#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "cuda_runtime.h"

class gpu_graph_t {
public:
    ~gpu_graph_t();

public:
    void add_kernel_node(const std::string& key, cudaKernelNodeParams params, cudaStream_t s);
    void update_kernel_node(const std::string& key, cudaKernelNodeParams params);
    void begin_capture(cudaStream_t s);
    void end_capture(cudaStream_t s);
    void launch_graph(cudaStream_t s);

private:
    std::unordered_map<std::string, cudaGraphNode_t> node_map_;
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    bool instantiated_{false};
};
