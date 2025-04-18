#pragma once

#include "common.cuh"

__host__ __device__ int mod(int a, int b);

void cpu_init(float* out, int w, int h);

__global__ void cuda_init(float* out, int w, int h);
__global__ void cuda_draw(float* out, int w, int h, int x, int y);

__host__ __device__ int idx(int x, int y, int w, int h);

// From https://www.shadertoy.com/view/XtdSDn
__host__ __device__ float sigmoid_a(float x, float a, float b);
__host__ __device__ float sigmoid_b(float x, float b, float eb);
__host__ __device__ float sigmoid_ab(float x, float a, float b, float ea, float eb);
__host__ __device__ float sigmoid_mix(float x, float y, float m, float em);
__host__ __device__ float trans_shadertoy(float inner, float outer);

// From Smooth Life paper
__host__ __device__ float sigmoid1(float x, float a, float alpha);
__host__ __device__ float sigmoid2(float x, float a, float b, float alpha);
__host__ __device__ float sigmoidm(float x, float y, float m, float alpha);
__host__ __device__ float trans_paper(float inner, float outer);

__device__ float saturate(float v);
__global__ void cuda_frame(float* in, float* out, int w, int h);
__global__ void cuda_val_to_col(float* in, unsigned char* out, int w, int h);

struct Config {
  // n - outer
  // m - inner

  // Values from https://www.shadertoy.com/view/XtdSDn
  float b1 = 0.257f;
  float b2 = 0.336f;
  float d1 = 0.365f;
  float d2 = 0.549f;
  float alpha_outer = 0.028f;
  float alpha_inner = 0.147f;
  float draw_value = 1.f;

  // Values SmoothLife paper
  // const float b1 = 0.278f;
  // const float b2 = 0.365f;
  // const float d1 = 0.267f;
  // const float d2 = 0.445f;
  // const float alpha_outer = 0.028f;
  // const float alpha_inner = 0.147f;

  float dt = 0.3f;
  // The algo is also slightly different
  bool is_paper;
};

class Cellular {
public:
  bool init(unsigned char* gpu_texture, int w, int h);
  bool term();
  bool update();
  bool draw(int x, int y);

private:
  unsigned char* gpu_texture = nullptr;
  int w;
  int h;

  GpuSlice<float> cuda_buf[2];
  CpuSlice<float> cpu_buf;

  const dim3 threads{16, 16};
  dim3 blocks;

  bool show_demo_window = false;

  Config config;
  Config* config_gpu;
};
