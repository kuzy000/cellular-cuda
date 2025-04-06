#pragma once

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "cellular.cuh"

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

  unsigned char* cuda_tex;
  const float fixed_fps = 60.f;

  const dim3 threads{16, 16};
  dim3 blocks;
  
  Cellular cellular;
};
