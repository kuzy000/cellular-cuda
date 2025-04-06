#pragma once

#include <stdio.h>

#define CUDA_CALL(f)                                                                            \
  do {                                                                                          \
    const cudaError_t res = f;                                                                  \
    if (res != cudaSuccess) {                                                                   \
      printf("ERROR: Calling " #f " %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res)); \
      return false;                                                                             \
    }                                                                                           \
  } while (false)

#define CUDA_KERNEL(...)                                                                                  \
  do {                                                                                                    \
    __VA_ARGS__;                                                                                          \
    const cudaError_t res = cudaPeekAtLastError();                                                        \
    if (res != cudaSuccess) {                                                                             \
      printf("ERROR: Calling " #__VA_ARGS__ " %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res)); \
      return false;                                                                                       \
    }                                                                                                     \
  } while (false)
