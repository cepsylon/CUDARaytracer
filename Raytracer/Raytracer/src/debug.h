#pragma once

#include <cuda_runtime.h>

void check_cuda_error(cudaError_t error, const char * func_name, const char * file, const int line);

#define CheckCUDAError(x) check_cuda_error((x), #x, __FILE__, __LINE__)
