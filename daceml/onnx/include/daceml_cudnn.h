#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>
#include <string>
#include <unordered_map>

namespace daceml {

    namespace cudnn {

        static void CheckCudnnError(cudnnStatus_t const& status) {
            if (status != CUDNN_STATUS_SUCCESS) {
                std::string error_str = "cuDNN failed with error: ";
                error_str.append(cudnnGetErrorString(status));
                printf("%s\n", error_str.c_str());
            }
        }

        static cudnnHandle_t CreateCudnnHandle(int device) {
            if (cudaSetDevice(device) != cudaSuccess) {
				printf("Failed to set CUDA device.\n");
            }
            cudnnHandle_t handle = nullptr;
            CheckCudnnError(cudnnCreate(&handle));
            return handle;
        }

/**
 * CUDNN wrapper class for DaCe. Once constructed, the class can be used to
 * get or create a CUDNN library handle (cudnnHandle_t) for a given GPU ID.

 * The class is constructed when the CUDNN DaCe library is used.
 **/
class CudnnHandle {
 public:

  cudnnHandle_t& Get(int device) {
    auto f = handles_.find(device);
    if (f == handles_.end()) {
      // Lazily construct new cuDNN handle if the specified key does not yet
      // exist
      auto handle = CreateCudnnHandle(device);
      f = handles_.emplace(device, handle).first;
    }
    return f->second;
  }

  ~CudnnHandle() {
    for (auto& h : handles_) {
      CheckCudnnError(cudnnDestroy(h.second));
    }
  }

//  CudnnHandle() = default;
//  CudnnHandle(CudnnHandle const&) = delete;
//  CudnnHandle& operator=(CudnnHandle const&) = delete;
 private:
  std::unordered_map<int, cudnnHandle_t> handles_;
};

    }  // namespace cudnn

}  // namespace daceml
