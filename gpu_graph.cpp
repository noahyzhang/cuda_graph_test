#include <iostream>
#include "cuda_helper.h"
#include "gpu_graph.h"

gpu_graph_t::~gpu_graph_t() {
    if (instantiated_) {
        cudaErrCheck(cudaGraphDestroy(graph_));
        cudaErrCheck(cudaGraphExecDestroy(graph_exec_));
        instantiated_ = false;
    }
}

void gpu_graph_t::add_kernel_node(const std::string& key, cudaKernelNodeParams params, cudaStream_t s) {
    cudaStreamCaptureStatus capture_status;
    cudaGraph_t graph;
    const cudaGraphNode_t* deps;
    size_t dep_count;
    cudaErrCheck(cudaStreamGetCaptureInfo_v2(s, &capture_status, nullptr, &graph, &deps, &dep_count));
    // 添加新的 node
    cudaGraphNode_t new_node;
    cudaErrCheck(cudaGraphAddKernelNode(&new_node, graph, deps, dep_count, &params));
    node_map_[key] = new_node;
    cudaErrCheck(cudaStreamUpdateCaptureDependencies(s, &new_node, 1, 1));
}

void gpu_graph_t::update_kernel_node(const std::string& key, cudaKernelNodeParams params) {
    cudaErrCheck(cudaGraphExecKernelNodeSetParams(graph_exec_, node_map_[key], &params));
}

void gpu_graph_t::begin_capture(cudaStream_t s) {
    cudaErrCheck(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
}

void gpu_graph_t::end_capture(cudaStream_t s) {
    if (instantiated_) {
        cudaErrCheck(cudaGraphDestroy(graph_));
    }
    cudaErrCheck(cudaStreamEndCapture(s, &graph_));
    bool need_instantiation = false;
    if (instantiated_) {
        cudaGraphExecUpdateResult update_result;
        cudaGraphExecUpdateResultInfo update_result_info;
        cudaErrCheck(cudaGraphExecUpdate(graph_exec_, graph_, &update_result_info));
        if (graph_exec_ == nullptr || update_result_info.result != cudaGraphExecUpdateSuccess) {
            // 更新不成功，需要重新初始化
            cudaGetLastError();
            if (graph_exec_ != nullptr) {
                cudaErrCheck(cudaGraphExecDestroy(graph_exec_));
            }
            need_instantiation = true;
        } else {
            need_instantiation = false;
        }
    } else {
        need_instantiation = true;
    }
    if (need_instantiation) {
        cudaErrCheck(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    }
    instantiated_ = true;
}

void gpu_graph_t::launch_graph(cudaStream_t s) {
    if (instantiated_) {
        cudaErrCheck(cudaGraphLaunch(graph_exec_, s));
    } else {
        std::cerr << "Launching an invalid or un_instantiated graph\n";
    }
}
