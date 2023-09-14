#pragma once

#include "cuda_runtime.h"
#include "stdio.h"

#define cudaErrCheck(stat)  \
{  \
    cuda_err_check_internal((stat), __FILE__, __LINE__);  \
}

inline void cuda_err_check_internal(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}
