#pragma once

#include "glad/gl.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

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

class App {
public:
  bool init();
  bool term();
  bool loop();

private:
  GLFWwindow* window = nullptr;

  // Screen quad
  GLuint buf_vert;

  GLuint shader_vert;
  GLuint shader_frag;
  GLuint shader_prog;

  GLuint texture;
  cudaGraphicsResource* tex_res;
  cudaArray* cuda_arr;

  const int w = 1024;
  const int h = 1024;

  const int tex_bytes = w * h * 4;
  const int size_bytes = w * h * sizeof(float);

  unsigned char* cuda_tex;
  float* cuda_buf[2];
  float* cpu_buf;

  const float fixed_fps = 60.f;
  int frame_number = 0;

  const dim3 threads{16, 16};
  dim3 blocks;
};
