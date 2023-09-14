#include <fstream>
#include <string>
#include "common.h"
#include "cuda_helper.h"
#include "inference.h"

int Inference::init(const std::string& engine_file, int buff_size) {
    // 将引擎数据文件导出
    char* trt_module_stream = nullptr;
    size_t size = 0;
    std::ifstream file(engine_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trt_module_stream = new char[size];
        assert(trt_module_stream);
        file.read(trt_module_stream, size);
        file.close();
    }
    // 创建引擎
    sample::Logger m_logger;
    runtime_ = nvinfer1::createInferRuntime(m_logger);
    assert(runtime_ != nullptr);
    engine_ = runtime_->deserializeCudaEngine(trt_module_stream, size, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);

    buffer_.resize(buff_size);
    return 0;
}

void Inference::release() {
    if (context_ != nullptr) {
        context_->destroy();
    }
    if (engine_ != nullptr) {
        engine_->destroy();
    }
    if (runtime_ != nullptr) {
        runtime_->destroy();
    }
}

void Inference::cuda_graph_inference() {
    cudaStream_t stream;
    cudaErrCheck(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&end));

    std::unordered_map<std::string, void*> in_map;
    cudaErrCheck(cudaMalloc(&in_map["input1"], BATCH_SIZE * 3 * IN_H * IN_W * sizeof(float)));
    cudaErrCheck(cudaMalloc(&in_map["input2"], BATCH_SIZE * 3 * IN_H * IN_W * sizeof(float)));

    std::unordered_map<std::string, void*> out_map;
    // cudaErrCheck(cudaMalloc(&out_map["input1"], BATCH_SIZE * 3 * IN_H * IN_W * sizeof(float)));
    // cudaErrCheck(cudaMalloc(&out_map["input2"], BATCH_SIZE * 3 * IN_H * IN_W * sizeof(float)));

    do_inference(stream, in_map, out_map, BATCH_SIZE);

    cudaErrCheck(cudaEventRecord(end, stream));
    cudaErrCheck(cudaEventSynchronize(end));
    float milli_sec = 0;
    cudaErrCheck(cudaEventElapsedTime(&milli_sec, start, end));
    fprintf(stdout, "Running with took: %6.2f ms\n", milli_sec);

    cudaErrCheck(cudaEventDestroy(start));
    cudaErrCheck(cudaEventDestroy(end));
    cudaErrCheck(cudaStreamDestroy(stream));
}

// __global__ void kernel_inference(
//     nvinfer1::IExecutionContext& context, std::vector<void*>& buffer,
//     cudaStream_t stream, int batch_size) {
//     context.enqueue(batch_size, buffer.data(), stream, nullptr);
//     cudaStreamSynchronize(stream);
// }

void Inference::do_inference(
    cudaStream_t stream,
    const std::unordered_map<std::string, void*>& in_map,
    const std::unordered_map<std::string, void*>& out_map,
    int batch_size) {

    // 开始捕获
    graph_.begin_capture(stream);

    cudaKernelNodeParams params;
    params.sharedMemBytes = 0;
    params.func = reinterpret_cast<void*>(kernel_inference);
    void* args[] = {context_, &buffer_, &stream, &batch_size};
    params.kernelParams = args;
    params.extra = nullptr;

    for (const auto& input : in_map) {
        const auto& name = input.first;
        const auto& addr = input.second;
        const int input_index = engine_->getBindingIndex(name.c_str());

        if (buffer_[input_index] != addr) {
            buffer_[input_index] = addr;

            graph_.add_kernel_node(name, params, stream);
        } else {
            graph_.update_kernel_node(name, params);
        }
    }
    // for (const auto& output : out_map) {
    //     const auto& name = output.first;
    //     const auto& addr = output.second;
    //     const int output_index = engine_->getBindingIndex(name.c_str());
    //     if (buffer_[output_index] != addr) {
    //         buffer_[output_index] = addr;
    //         graph_.add_kernel_node(name, params, stream);
    //     } else {
    //         graph_.update_kernel_node(name, params);
    //     }
    // }

    context_->enqueue(batch_size, buffer_.data(), stream, nullptr);
    cudaStreamSynchronize(stream);

    graph_.end_capture(stream);
}
