#include "onnxruntime_c_api.h"

#ifndef __DACE_ONNX_CUDA_H
#define __DACE_ONNX_CUDA_H
OrtMemoryInfo* __ort_cuda_mem_info;
OrtMemoryInfo* __ort_cuda_pinned_mem_info;
#endif  // __DACE_ONNX_CUDA_H