#pragma once

__host__ __device__ int mod(int a, int b);

void cpu_init(float* out, int w, int h);

__global__ void cuda_init(float* out, int w, int h);

__host__ __device__ int idx(int x, int y, int w, int h);

// From https://www.shadertoy.com/view/XtdSDn#
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

class Cellular {
public:
  bool init(unsigned char* gpu_texture, int w, int h);
  bool term();
  bool update();

private:
  unsigned char* gpu_texture = nullptr;
  int w;
  int h;

  float* cuda_buf[2];
  float* cpu_buf;

  const dim3 threads{16, 16};
  dim3 blocks;
  
  bool show_demo_window = false;
  
  struct Config {
      float dt = 0.3f;
  };

  Config config;
};
