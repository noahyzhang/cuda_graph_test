#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "NvInfer.h"
#include "../../TensorRT/samples/common/logger.h"
#include "gpu_graph.h"

class Inference {
public:
    int init(const std::string& engine_file, int buff_size);
    void release();
    void cuda_graph_inference();

    void do_inference(
        cudaStream_t stream,
        const std::unordered_map<std::string, void*>& in_map,
        const std::unordered_map<std::string, void*>& out_map,
        int batch_size);

private:
    void do_inference_internal(int batch_size);

private:
    std::vector<void*> buffer_;
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    gpu_graph_t graph_;
};
