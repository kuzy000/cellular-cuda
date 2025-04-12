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
  void on_mouse_button(int button, int action, int mods);
  void on_scroll(float x, float y);
  void on_cursor(float x, float y);

  GLFWwindow* window = nullptr;

  // Screen quad
  GLuint buf_vert;

  GLuint shader_vert;
  GLuint shader_frag;
  GLuint shader_prog;

  GLuint texture;
  cudaGraphicsResource* tex_res;
  cudaArray* cuda_arr;

  const int tex_w = 1024;
  const int tex_h = 1024;
  int win_w = 1024;
  int win_h = 1024;

  unsigned char* cuda_tex;
  const float fixed_fps = 60.f;

  const dim3 threads{16, 16};
  dim3 blocks;

  float offset_x = 0.f;
  float offset_y = 0.f;
  float scale = 1.f;
  bool is_drag = false;
  bool is_draw = false;
  float drag_x = 0.f;
  float drag_y = 0.f;
  float scx = 1.f;
  float scy = 1.f;

  float cursor_x = 0.f;
  float cursor_y = 0.f;

  Cellular cellular;
};
