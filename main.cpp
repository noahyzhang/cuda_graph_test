#include "gpu_graph.h"
#include "inference.h"
#include "module.h"

#define MODEL_ENGINE_NAME "model_test.engine"

int test_01() {
    std::unordered_map<std::string, std::string> in_out;
    in_out["input1"] = "output1";
    in_out["input2"] = "output2";
    int res = Module::build_module(MODEL_ENGINE_NAME, in_out);
    if (res < 0) {
        std::cerr << "build module failed" << std::endl;
        return -1;
    }

    Inference inference_obj;
    if (inference_obj.init(MODEL_ENGINE_NAME, in_out.size()) < 0) {
        std::cerr << "inference init failed" << std::endl;
        return -2;
    }
    inference_obj.cuda_graph_inference();

    inference_obj.release();
}

int main() {
    test_01();
    return 0;
}
