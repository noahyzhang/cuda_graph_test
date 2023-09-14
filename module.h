#pragma once

#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include "NvInfer.h"
#include "../../TensorRT/samples/common/logger.h"
#include "common.h"

class Module {
public:
    static int build_module(
        const std::string& engin_file_name,
        const std::unordered_map<std::string, std::string>& in_out) {
        sample::Logger m_logger;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(m_logger);

        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 20);
        // 创建网络
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH);
        // 创建引擎
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

        for (const auto& x : in_out) {
            profile->setDimensions(x.first.c_str(), nvinfer1::OptProfileSelector::kMIN,
                nvinfer1::Dims4{BATCH_SIZE, 3, IN_H, IN_W});
            profile->setDimensions(x.first.c_str(), nvinfer1::OptProfileSelector::kOPT,
                nvinfer1::Dims4{BATCH_SIZE, 3, IN_H, IN_W});
            profile->setDimensions(x.first.c_str(), nvinfer1::OptProfileSelector::kMAX,
                nvinfer1::Dims4{BATCH_SIZE, 3, IN_H, IN_W});

            // 添加 input 和 output
            nvinfer1::ITensor* input_tensor = network->addInput(
                x.first.c_str(), nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{BATCH_SIZE, 3, IN_H, IN_W});
            nvinfer1::IPoolingLayer* pool = network->addPoolingNd(
                *input_tensor, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
            pool->setStrideNd(nvinfer1::DimsHW{2, 2});
            pool->getOutput(0)->setName(x.second.c_str());

            network->markOutput(*pool->getOutput(0));
        }
        nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        nvinfer1::IHostMemory* model_stream{nullptr};
        assert(engine != nullptr);
        model_stream = engine->serialize();
        // 写文件
        std::ofstream p(engin_file_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open output file to save model" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
        std::cout << "generating model file success" << std::endl;

        // 析构
        model_stream->destroy();
        network->destroy();
        config->destroy();
        engine->destroy();
        builder->destroy();
        return 0;
    }
};
