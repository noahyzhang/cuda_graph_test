#pragma once

#include "NvInfer.h"

const int IN_H = 224;
const int IN_W = 224;
const int BATCH_SIZE = 1;
const int EXPLICIT_BATCH = 1 << (int)(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
