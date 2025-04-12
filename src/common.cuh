#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CALL(f)                                                                                \
  do {                                                                                              \
    const cudaError_t __res = f;                                                                    \
    if (__res != cudaSuccess) {                                                                     \
      printf("ERROR: Calling " #f " %s: %s\n", cudaGetErrorName(__res), cudaGetErrorString(__res)); \
      return false;                                                                                 \
    }                                                                                               \
  } while (false)

#define CUDA_ABORT(f)                                                                               \
  do {                                                                                              \
    const cudaError_t __res = f;                                                                    \
    if (__res != cudaSuccess) {                                                                     \
      printf("ERROR: Calling " #f " %s: %s\n", cudaGetErrorName(__res), cudaGetErrorString(__res)); \
      abort();                                                                                      \
    }                                                                                               \
  } while (false)

#define CUDA_KERNEL(...)                                                                                      \
  do {                                                                                                        \
    __VA_ARGS__;                                                                                              \
    const cudaError_t __res = cudaPeekAtLastError();                                                          \
    if (__res != cudaSuccess) {                                                                               \
      printf("ERROR: Calling " #__VA_ARGS__ " %s: %s\n", cudaGetErrorName(__res), cudaGetErrorString(__res)); \
      return false;                                                                                           \
    }                                                                                                         \
  } while (false)

template <typename T>
T* alloc_cpu(size_t len = 1) {
  return (T*)malloc(sizeof(T) * len);
}

template <typename T>
T* alloc_gpu(size_t len = 1) {
  void* res = nullptr;
  CUDA_ABORT(cudaMalloc(&res, sizeof(T) * len));
  return (T*)res;
}

template <typename T>
void copy_to_cpu(T* cpu_dst, T* gpu_src, size_t len = 1) {
  CUDA_ABORT(cudaMemcpy(cpu_dst, gpu_src, sizeof(T) * len, cudaMemcpyDeviceToHost));
}

template <typename T>
void copy_to_gpu(T* gpu_dst, T* cpu_src, size_t len = 1) {
  CUDA_ABORT(cudaMemcpy(gpu_dst, cpu_src, sizeof(T) * len, cudaMemcpyHostToDevice));
}

template <typename T>
struct CpuSlice;

template <typename T>
struct GpuSlice;

template <typename T>
struct GpuSlice {
  T* ptr;
  size_t len;
  
  static GpuSlice<T> alloc(size_t len) {
    return GpuSlice<T> {alloc_gpu<T>(len), len};
  }
  
  void copy_from(GpuSlice<T> val) {
    assert(val.len == len);
    CUDA_ABORT(cudaMemcpy(ptr, val.ptr, sizeof(T) * len, cudaMemcpyDeviceToDevice));
  }

  void copy_from(CpuSlice<T> val) {
    assert(val.len == len);
    CUDA_ABORT(cudaMemcpy(ptr, val.ptr, sizeof(T) * len, cudaMemcpyHostToDevice));
  }
};

template <typename T>
struct CpuSlice {
  T* ptr;
  size_t len;

  static CpuSlice<T> alloc(size_t len) {
    return CpuSlice<T> {alloc_cpu<T>(len), len};
  }

  void copy_from(GpuSlice<T> val) {
    assert(val.len == len);
    CUDA_ABORT(cudaMemcpy(ptr, val.ptr, sizeof(T) * len, cudaMemcpyDeviceToHost));
  }

  void copy_from(CpuSlice<T> val) {
    assert(val.len == len);
    CUDA_ABORT(cudaMemcpy(ptr, val.ptr, sizeof(T) * len, cudaMemcpyHostToHost));
  }
};