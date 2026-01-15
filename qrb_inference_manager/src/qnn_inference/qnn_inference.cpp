// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "qnn_inference/qnn_inference.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <unordered_map>

#include "QnnContext.h"
#include "lib_mem_dmabuf/dmabuf.hpp"

namespace qrb::inference_mgr
{

// callback notifyParam：为避免访问 QnnInference private 嵌套类型导致编译失败，
// 该结构体定义为本 .cpp 文件的“私有实现细节”，并由 QnnInference 以 void* 形式持有。
struct ContextBinaryCallbackCtx {
  int model_fd = -1;

  struct DmaAllocation {
    std::shared_ptr<lib_mem_dmabuf::DmaBuffer> buf;
    uint64_t aligned_size = 0;
  };

  // key: dma fd（QNN 回调要求每次返回 distinct fd）
  std::unordered_map<int, DmaAllocation> dmabufs;

  // ---- verification / instrumentation ----
  uint64_t total_allocated_bytes = 0;   // sum(aligned_size) over provider calls
  uint64_t max_single_alloc_bytes = 0;  // max(aligned_size)
  uint32_t num_allocs = 0;              // number of provider calls
  uint64_t total_requested_bytes = 0;   // sum(req.size)
};

namespace {

static constexpr uint64_t kPageSize = 4096;
static inline uint64_t align_up(uint64_t v, uint64_t align) {
  return (v + align - 1) / align * align;
}

static constexpr const char* kDefaultDmaHeap = "/dev/dma_heap/system";

static bool allocate_dmabuf(uint64_t size,
                            ContextBinaryCallbackCtx::DmaAllocation* outAlloc,
                            Qnn_MemDmaBufInfo_t* outQnnDmaBuf) {
  if (!outAlloc || !outQnnDmaBuf || size == 0) {
    return false;
  }

  const uint64_t alignedSize = align_up(size, kPageSize);

  auto buf = lib_mem_dmabuf::DmaBuffer::alloc(alignedSize, kDefaultDmaHeap);
  if (!buf) {
    return false;
  }
  if (!buf->map()) {
    return false;
  }

  outAlloc->buf = buf;
  outAlloc->aligned_size = alignedSize;

  outQnnDmaBuf->fd = buf->fd();
  outQnnDmaBuf->data = buf->addr();
  return true;
}

static void free_dmabuf(ContextBinaryCallbackCtx::DmaAllocation& alloc) {
  if (!alloc.buf) {
    return;
  }
  // DmaBuffer::~DmaBuffer 会在 auto_release_==true 时 close(fd) + unmap
  alloc.buf.reset();
  alloc.aligned_size = 0;
}


/*
1. 申请DMA buffer并mmap成CPU地址
2. 将model.bin中的数据读入
*/
static Qnn_ErrorHandle_t dma_data_provider(Qnn_ContextBinaryDataRequest_t req,
                                          Qnn_ContextBinaryDmaDataResponse_t* resp,
                                          void* notifyParam) {
  if (!resp || !notifyParam || req.size == 0) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  auto* ctx = reinterpret_cast<ContextBinaryCallbackCtx*>(notifyParam);

  const uint64_t alignedSize = align_up(static_cast<uint64_t>(req.size), kPageSize);

  Qnn_MemDmaBufInfo_t dmaBuf{}; //存放了DMA Buffer fd
  ContextBinaryCallbackCtx::DmaAllocation alloc{};
  if (!allocate_dmabuf(alignedSize, &alloc, &dmaBuf) || dmaBuf.fd < 0 || dmaBuf.data == nullptr) {
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }

  // Instrumentation: prove DMA-BUF allocation succeeded and record sizes.
  // - dmaBuf.fd: should be a dmabuf fd (often shows as anon_inode:dmabuf in /proc/<pid>/fd)
  // - dmaBuf.data: mmap'ed CPU-visible address
  // - alloc.aligned_size: actual allocated bytes (>= req.size, 4KB aligned)
  ctx->num_allocs++;
  ctx->total_requested_bytes += static_cast<uint64_t>(req.size);
  ctx->total_allocated_bytes += alloc.aligned_size;
  if (alloc.aligned_size > ctx->max_single_alloc_bytes) {
    ctx->max_single_alloc_bytes = alloc.aligned_size;
  }

  QRB_DEBUG("[QNN-DMABUF] provider: req{off=",
            static_cast<uint64_t>(req.offset),
            ",size=",
            static_cast<uint64_t>(req.size),
            ",map=",
            static_cast<unsigned>(req.isBackendMappingNeeded),
            "} -> dmabuf{fd=",
            dmaBuf.fd,
            ",addr=",
            dmaBuf.data,
            ",alloc=",
            static_cast<uint64_t>(alloc.aligned_size),
            "}");

  // 按照QNN SDK文档的建议，令dataStartOffset=0，避免后端 mapping 对 offset 的限制
  const uint64_t dataStartOffset = 0;

  // 同步把 binary 的指定分段读到 DMA buffer
  ssize_t bytesRead = ::pread(ctx->model_fd,
                              static_cast<uint8_t*>(dmaBuf.data) + dataStartOffset,
                              static_cast<size_t>(req.size),
                              static_cast<off_t>(req.offset));
  if (bytesRead != static_cast<ssize_t>(req.size)) {
    free_dmabuf(alloc);
    return QNN_CONTEXT_ERROR_CREATE_FROM_BINARY;
  }

  resp->dmaBuffer = dmaBuf;
  resp->dataStartOffset = static_cast<Qnn_ContextBinarySize_t>(dataStartOffset);
  resp->alignedSize = static_cast<Qnn_ContextBinarySize_t>(alloc.aligned_size);

  ctx->dmabufs.emplace(dmaBuf.fd, std::move(alloc));
  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t dma_data_release(Qnn_ContextBinaryDmaDataMem_t mem, void* notifyParam) {
  if (!notifyParam) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  auto* ctx = reinterpret_cast<ContextBinaryCallbackCtx*>(notifyParam);
  const int fd = mem.dmaBuffer.fd;

  QRB_DEBUG("[QNN-DMABUF] release: dmabuf{fd=",
            fd,
            ",memSize=",
            static_cast<uint64_t>(mem.memSize),
            "} (tracked=",
            ctx->dmabufs.size(),
            ")");

  auto it = ctx->dmabufs.find(fd);
  if (it == ctx->dmabufs.end()) {
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  free_dmabuf(it->second);
  ctx->dmabufs.erase(it);
  return QNN_SUCCESS;
}

}  // namespace

QnnInference::QnnInference(const std::string & model_path, const std::string & backend_option)
  : model_path_(model_path), backend_option_(backend_option)
{
  auto is_bin_model = (std::string::npos != model_path.find(".bin"));

  if (is_bin_model) {
    QRB_INFO("Loading model from binary file: ", model_path);

    load_model_from_binary = true;
    qnn_interface_ = std::make_unique<QnnInterface>(
        backend_option_, &backend_lib_handle, qnn_syslib_path_, &sys_lib_handle_);
  } else {
    qnn_interface_ = std::make_unique<QnnInterface>(
        model_path_, backend_option_, &backend_handle_, &model_handle_);
  }
}

QnnInference::~QnnInference()
{
  free_graphs_info();
  free_context();
  free_device();
  free_backend();

  if (backend_lib_handle) {
    ::dlclose(backend_lib_handle);
    backend_lib_handle = nullptr;
  }

  if (model_handle_) {
    ::dlclose(model_handle_);
    model_handle_ = nullptr;
  }

  if (sys_lib_handle_) {
    ::dlclose(sys_lib_handle_);
    sys_lib_handle_ = nullptr;
  }
}

StatusCode QnnInference::inference_init()
{
  if (initialize_backend() != StatusCode::SUCCESS) {
    return StatusCode::FAILURE;
  }

  if (create_device() != StatusCode::SUCCESS) {
    return StatusCode::FAILURE;
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnInference::inference_graph_init()
{
  if (true == load_model_from_binary) {
    if (init_graph_from_binary() != StatusCode::SUCCESS) {
      return StatusCode::FAILURE;
    }
  } else {
    if (create_context() != StatusCode::SUCCESS) {
      return StatusCode::FAILURE;
    }

    if (compose_graphs() != StatusCode::SUCCESS) {
      return StatusCode::FAILURE;
    }

    if (finalize_graphs() != StatusCode::SUCCESS) {
      return StatusCode::FAILURE;
    }
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnInference::inference_execute(const std::vector<uint8_t> & input_tensor_data)
{
  if (input_tensor_data.size() == 0) {
    QRB_ERROR("Input tensor is NULL!");
    return StatusCode::FAILURE;
  }

  for (uint32_t i = 0; i < graphs_count_; i++) {
    const auto & graphs_info = (*(graphs_info_))[i];

    auto io_tensors =
        QnnTensor(graphs_info.num_of_input_tensors, graphs_info.num_of_output_tensors);

    if (StatusCode::SUCCESS != io_tensors.setup_tensors(io_tensors.inputs,
                                   io_tensors.num_of_input_tensors, graphs_info.input_tensors)) {
      QRB_ERROR("Setup input tensors failed!");
      return StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS != io_tensors.setup_tensors(io_tensors.outputs,
                                   io_tensors.num_of_output_tensors, graphs_info.output_tensors)) {
      QRB_ERROR("Setup output tensors failed!");
      return StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS != io_tensors.write_input_tensors(input_tensor_data)) {
      QRB_ERROR("Write QNN input tensors failed!");
      return StatusCode::FAILURE;
    }

    if (QNN_GRAPH_NO_ERROR != this->qnn_interface_->interface.graphExecute(graphs_info.graph,
                                  io_tensors.inputs, io_tensors.num_of_input_tensors,
                                  io_tensors.outputs, io_tensors.num_of_output_tensors, nullptr,
                                  nullptr)) {
      QRB_ERROR("QNN graphExecute failed!");
      return StatusCode::FAILURE;
    }

#ifndef __hexagon__
    output_tensor_ = io_tensors.read_output_tensors(graphs_info.num_of_output_tensors);
    if (output_tensor_.size() == 0) {
      QRB_ERROR("Get ouput tensors failed!");
    }
#endif
  }

  return StatusCode::SUCCESS;
}

const std::vector<OutputTensor> QnnInference::get_output_tensors()
{
  return std::move(output_tensor_);
}

StatusCode QnnInference::initialize_backend()
{
  auto qnn_status = qnn_interface_->interface.backendCreate(
      nullptr, (const QnnBackend_Config_t **)(nullptr), &(backend_handle_));

  if (QNN_BACKEND_NO_ERROR != qnn_status) {
    QRB_ERROR("Could not initialize backend due to error = %d!", qnn_status);
    return StatusCode::FAILURE;
  }

  QRB_INFO(backend_option_, " initialize successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::create_device()
{
  auto is_device_property_supported = [this] {
    if (nullptr != qnn_interface_->interface.propertyHasCapability) {
      auto qnn_status = qnn_interface_->interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);

      if (QNN_PROPERTY_NOT_SUPPORTED == qnn_status) {
        QRB_WARNING("Device property is not supported!");
      } else if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnn_status) {
        QRB_ERROR("Device property is not known to backend!");
        return StatusCode::FAILURE;
      }
    }
    return StatusCode::SUCCESS;
  };

  if (StatusCode::FAILURE != is_device_property_supported()) {
    if (nullptr != qnn_interface_->interface.deviceCreate) {
      auto qnn_status = qnn_interface_->interface.deviceCreate(nullptr, nullptr, &(device_handle_));

      if (QNN_SUCCESS != qnn_status && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnn_status) {
        QRB_ERROR("Failed to create device!");
        return StatusCode::FAILURE;
      }
    }
    support_device_ = true;
  }

  QRB_INFO("Qnn device initialize successfully");
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::create_context()
{
  if (QNN_CONTEXT_NO_ERROR != qnn_interface_->interface.contextCreate(
                                  backend_handle_, device_handle_, nullptr, &(context_))) {
    QRB_ERROR("Could not create context!");
    return StatusCode::FAILURE;
  }
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::compose_graphs()
{
  if (ModelError::MODEL_NO_ERROR !=
      qnn_interface_->compose_graphs(backend_handle_, qnn_interface_->interface, context_, nullptr,
          0, &(graphs_info_), &(graphs_count_), false, nullptr, QNN_LOG_LEVEL_MAX)) {
    QRB_ERROR("Failed in composeGraphs()!");
    return StatusCode::FAILURE;
  }
  return StatusCode::SUCCESS;
}

StatusCode QnnInference::finalize_graphs()
{
  for (size_t i = 0; i < graphs_count_; i++) {
    if (QNN_GRAPH_NO_ERROR !=
        qnn_interface_->interface.graphFinalize((*graphs_info_)[i].graph, nullptr, nullptr)) {
      return StatusCode::FAILURE;
    }
  }

  return StatusCode::SUCCESS;
}

void QnnInference::free_graphs_info()
{
  if (graphs_info_ == nullptr) {
    return;
  }

  for (uint32_t i = 0; i < graphs_count_; i++) {
    auto & graph_info = graphs_info_[i];
    free(graph_info->graph_name);
    graph_info->graph_name = nullptr;

    QnnTensor tensor_ops;
    tensor_ops.free_qnn_tensors(graph_info->input_tensors, graph_info->num_of_input_tensors);
    tensor_ops.free_qnn_tensors(graph_info->output_tensors, graph_info->num_of_output_tensors);
  }

  free(*graphs_info_);
  *graphs_info_ = nullptr;
  free(graphs_info_);
  graphs_info_ = nullptr;
}

void QnnInference::free_context()
{
  if (context_ != nullptr) {
    if (QNN_CONTEXT_NO_ERROR != qnn_interface_->interface.contextFree(context_, nullptr)) {
      QRB_ERROR("Failed to free context!");
    }
    context_ = nullptr;
  }

  // 释放 callback 相关资源（仅 binary 模型 + HTP + WithCallback 路径会分配）
  if (context_binary_cb_ctx_) {
    auto* ctx = reinterpret_cast<ContextBinaryCallbackCtx*>(context_binary_cb_ctx_.get());

    if (ctx->model_fd >= 0) {
      ::close(ctx->model_fd);
      ctx->model_fd = -1;
    }

    // 正常情况下 QNN 会在 contextFree 过程中触发 dma_data_release，把 dmabufs 清空；
    // 这里保留兜底，防止异常路径泄漏。
    for (auto& kv : ctx->dmabufs) {
      free_dmabuf(kv.second);
    }
    ctx->dmabufs.clear();
    context_binary_cb_ctx_.reset();
  }
}

void QnnInference::free_device()
{
  if (true == support_device_) {
    if (nullptr != qnn_interface_->interface.deviceFree) {
      auto qnn_status = qnn_interface_->interface.deviceFree(device_handle_);
      if (QNN_SUCCESS != qnn_status && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnn_status) {
        QRB_ERROR("Failed to free device!");
      }
    }
  }

  device_handle_ = nullptr;
}

void QnnInference::free_backend()
{
  if (qnn_interface_->interface.backendFree != nullptr) {
    if (QNN_BACKEND_NO_ERROR != qnn_interface_->interface.backendFree(backend_handle_)) {
      QRB_ERROR("Could not free backend!");
    }
  }

  backend_handle_ = nullptr;
}

StatusCode QnnInference::init_graph_from_binary()
{
  auto [model_buf, model_buf_size] = read_binary_model();
  if (nullptr == model_buf) {
    return StatusCode::FAILURE;
  }

  if (StatusCode::SUCCESS != get_and_set_graph_info_from_binary(model_buf, model_buf_size)) {
    return StatusCode::FAILURE;
  }

  if (StatusCode::SUCCESS != create_context_from_binary(model_buf, model_buf_size)) {
    return StatusCode::FAILURE;
  }

  QRB_INFO("Initialize Qnn graph from binary file successfully");
  return StatusCode::SUCCESS;
}

std::tuple<std::shared_ptr<uint8_t[]>, uint64_t> QnnInference::read_binary_model()
{
  uint64_t buf_size = 0;
  std::ifstream model_binary(model_path_, std::ifstream::binary);
  if (!model_binary.is_open() || model_binary.fail()) {
    QRB_ERROR("Fail to open model file: ", model_path_);
    return { nullptr, 0 };
  }

  model_binary.seekg(0, model_binary.end);
  buf_size = model_binary.tellg();
  model_binary.seekg(0, model_binary.beg);
  if (0 == buf_size) {
    QRB_ERROR("Received path to an empty file. Nothing to deserialize.");
    return { nullptr, 0 };
  }

  std::shared_ptr<uint8_t[]> model_buf(new uint8_t[buf_size]);
  if (!model_binary.read(reinterpret_cast<char *>(model_buf.get()), buf_size)) {
    QRB_ERROR("Fail to read model file: ", model_path_);
    return { nullptr, 0 };
  }

  return { model_buf, buf_size };
}

StatusCode QnnInference::get_and_set_graph_info_from_binary(
    const std::shared_ptr<uint8_t[]> model_buf,
    const uint64_t model_buf_size)
{
  QnnSystemContext_Handle_t sys_context_headle = nullptr;
  if (QNN_SUCCESS !=
      qnn_interface_->qnn_system_interface.systemContextCreate(&sys_context_headle)) {
    QRB_ERROR("Could not create system handle.");
    return StatusCode::FAILURE;
  }

  const QnnSystemContext_BinaryInfo_t * binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size = 0;
  if (QNN_SUCCESS !=
      qnn_interface_->qnn_system_interface.systemContextGetBinaryInfo(sys_context_headle,
          static_cast<void *>(model_buf.get()), model_buf_size, &binary_info, &binary_info_size)) {
    QRB_ERROR("Failed to get context binary info.");
    return StatusCode::FAILURE;
  }

  if (StatusCode::SUCCESS != set_up_graph_info(binary_info)) {
    return StatusCode::FAILURE;
  }

  qnn_interface_->qnn_system_interface.systemContextFree(sys_context_headle);
  sys_context_headle = nullptr;

  return StatusCode::SUCCESS;
}

template <typename T>
StatusCode QnnInference::copy_graph_info(T graph_info_from_binary)
{
  graphs_info_ = (GraphInfo **)calloc(graphs_count_, sizeof(GraphInfo *));
  if (nullptr == graphs_info_) {
    return StatusCode::FAILURE;
  }

  for (uint32_t i = 0; i < graphs_count_; i++) {
    auto graph_info_dst = (GraphInfo *)calloc(1, sizeof(GraphInfo));
    if (nullptr == graph_info_dst) {
      return StatusCode::FAILURE;
    }

    graph_info_dst->graph_name =
        (char *)malloc(sizeof(char) * strlen(graph_info_from_binary.graphName));
    if (nullptr == graph_info_dst->graph_name) {
      return StatusCode::FAILURE;
    }
    memcpy(graph_info_dst->graph_name, graph_info_from_binary.graphName,
        strlen(graph_info_from_binary.graphName));

    auto set_up_tensors_info = [](const Qnn_Tensor_t * src, const uint32_t cnt,
                                   Qnn_Tensor_t *& dst) {
      dst = (Qnn_Tensor_t *)calloc(cnt, sizeof(Qnn_Tensor_t));
      if (nullptr == dst) {
        return StatusCode::FAILURE;
      }

      QnnTensor tensor_ops;

      for (size_t i = 0; i < cnt; i++) {
        dst[i] = QNN_TENSOR_INIT;
        if (StatusCode::SUCCESS != tensor_ops.tensor_info_deep_copy(&dst[i], &src[i])) {
          return StatusCode::FAILURE;
        }
      }

      return StatusCode::SUCCESS;
    };

    graph_info_dst->num_of_input_tensors = graph_info_from_binary.numGraphInputs;
    if (StatusCode::SUCCESS != set_up_tensors_info(graph_info_from_binary.graphInputs,
                                   graph_info_from_binary.numGraphInputs,
                                   graph_info_dst->input_tensors)) {
      return StatusCode::FAILURE;
    }

    graph_info_dst->num_of_output_tensors = graph_info_from_binary.numGraphOutputs;
    if (StatusCode::SUCCESS != set_up_tensors_info(graph_info_from_binary.graphOutputs,
                                   graph_info_from_binary.numGraphOutputs,
                                   graph_info_dst->output_tensors)) {
      return StatusCode::FAILURE;
    }

    graphs_info_[i] = graph_info_dst;
  }

  return StatusCode::SUCCESS;
}

/// @brief set up graphs_info_ based on binary_info
/// @param binary_info
/// @return SUCCESS or FAILURE
StatusCode QnnInference::set_up_graph_info(const QnnSystemContext_BinaryInfo_t * binary_info)
{
  if (nullptr == binary_info) {
    return StatusCode::FAILURE;
  }

  switch (binary_info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
      graphs_count_ = binary_info->contextBinaryInfoV1.numGraphs;
      if (StatusCode::SUCCESS !=
          copy_graph_info(binary_info->contextBinaryInfoV1.graphs->graphInfoV1)) {
        return StatusCode::FAILURE;
      }
      break;
    }
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
      graphs_count_ = binary_info->contextBinaryInfoV2.numGraphs;
      if (StatusCode::SUCCESS !=
          copy_graph_info(binary_info->contextBinaryInfoV2.graphs->graphInfoV2)) {
        return StatusCode::FAILURE;
      }
      break;
    }
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
      graphs_count_ = binary_info->contextBinaryInfoV3.numGraphs;
      if (StatusCode::SUCCESS !=
          copy_graph_info(binary_info->contextBinaryInfoV3.graphs->graphInfoV3)) {
        return StatusCode::FAILURE;
      }
      break;
    }
    default: {
      QRB_ERROR("Unrecognized system context binary info version.");
      return StatusCode::FAILURE;
    }
  }

  return StatusCode::SUCCESS;
}

StatusCode QnnInference::create_context_from_binary(const std::shared_ptr<uint8_t[]> model_buf,
    const uint64_t model_buf_size)
{
  const bool is_htp_backend = (backend_option_.find("HTP") != std::string::npos);

  if (!is_htp_backend || !enable_context_create_callback_|| qnn_interface_->interface.contextCreateFromBinaryWithCallback == nullptr) {
    if (qnn_interface_->interface.contextCreateFromBinary(backend_handle_, device_handle_, nullptr,
            static_cast<void *>(model_buf.get()), model_buf_size, &context_, nullptr)) {
      QRB_ERROR("Could not create context from binary!");
      return StatusCode::FAILURE;
    }
  } else {
    // HTP 优化路径：createFromBinaryWithCallback（external weights-loaded buffer）
    // NOTE:
    // - callback 依赖 DMA-BUF allocator；本仓库不引入平台 allocator，实现留在 allocate_dmabuf()
    // - notifyParam 必须在 contextFree 前保持有效，因此保存到成员 context_binary_cb_ctx_
    auto* rawCtx = new ContextBinaryCallbackCtx();
    rawCtx->model_fd = ::open(model_path_.c_str(), O_RDONLY);
    if (rawCtx->model_fd < 0) {
      QRB_ERROR("Fail to open model file for callback provider: ", model_path_);
      delete rawCtx;
      return StatusCode::FAILURE;
    }

    // 用 unique_ptr<void,deleter> 持有，保证 notifyParam 生命周期覆盖 contextFree。
    context_binary_cb_ctx_ = std::unique_ptr<void, void (*)(void*)>(
        rawCtx, [](void* p) { delete reinterpret_cast<ContextBinaryCallbackCtx*>(p); });

    Qnn_ContextBinaryCallback_t callback{};
    callback.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
    callback.dmaBufferCallback.version = QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
    callback.dmaBufferCallback.v1.dataProvide = dma_data_provider;
    callback.dmaBufferCallback.v1.dataRelease = dma_data_release;
    callback.dmaBufferCallback.v1.notifyParam = rawCtx;

    Qnn_SignalHandle_t signal = nullptr;
    const Qnn_ErrorHandle_t err = qnn_interface_->interface.contextCreateFromBinaryWithCallback(
        backend_handle_,
        device_handle_,
        nullptr,
        &callback,
        static_cast<void *>(model_buf.get()),
        model_buf_size,
        &context_,
        nullptr,
        signal);

    if (err != QNN_SUCCESS) {
      // create 失败时，QNN 未必触发 release 回调：做兜底释放
      auto* failCtx = reinterpret_cast<ContextBinaryCallbackCtx*>(context_binary_cb_ctx_.get());
      for (auto& kv : failCtx->dmabufs) {
        free_dmabuf(kv.second);
      }
      failCtx->dmabufs.clear();
      ::close(failCtx->model_fd);

      QRB_DEBUG("[QNN-DMABUF] create failed: reqTotal=",
                static_cast<uint64_t>(failCtx->total_requested_bytes),
                " allocTotal=",
                static_cast<uint64_t>(failCtx->total_allocated_bytes),
                " maxAlloc=",
                static_cast<uint64_t>(failCtx->max_single_alloc_bytes),
                " numAllocs=",
                static_cast<unsigned>(failCtx->num_allocs));

      context_binary_cb_ctx_.reset();

      QRB_ERROR("Could not create context from binary with callback! err=", err);
      return StatusCode::FAILURE;
    }

    // create 成功后打印统计信息，便于验证：
    // 1) 是否真的分配了 DMA-BUF（numAllocs>0 且 fd/addr 在 provider 日志中出现）
    // 2) 分配总量/最大单块是多少
    QRB_DEBUG("[QNN-DMABUF] create ok: reqTotal=",
              static_cast<uint64_t>(rawCtx->total_requested_bytes),
              " allocTotal=",
              static_cast<uint64_t>(rawCtx->total_allocated_bytes),
              " maxAlloc=",
              static_cast<uint64_t>(rawCtx->max_single_alloc_bytes),
              " numAllocs=",
              static_cast<unsigned>(rawCtx->num_allocs));
  }

  for (uint32_t i = 0; i < graphs_count_; i++) {
    if (QNN_SUCCESS != qnn_interface_->interface.graphRetrieve(
                           context_, (*graphs_info_)[i].graph_name, &((*graphs_info_)[i].graph))) {
      QRB_ERROR("Unable to retrieve graph handle for the graph");
      return StatusCode::FAILURE;
    }
  }
  return StatusCode::SUCCESS;
}

}  // namespace qrb::inference_mgr
