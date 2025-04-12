#include "cellular.cuh"
#include "imgui.h"
#include <cstdio>
#include <cstdlib>

constexpr float PI = 3.141592654f;

__host__ __device__ int mod(int a, int b) {
  const int r = a % b;
  return r >= 0 ? r : r + b;
}

void cpu_init(float* out, int w, int h) {
  memset(out, 0, w * h * 4);

  for (int i = 0; i < w * h; ++i) {
    float x = i % w;
    float y = int(i / w);

    x /= w;
    y /= h;

    out[i] = x * y;
  }
}

__global__ void cuda_init(float* out, int w, int h) {
  return;
}

__global__ void cuda_draw(float* out, int w, int h, int px, int py, float draw_val) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const float dx = (x - px) / (float)w;
  const float dy = (y - py) / (float)h;

  const float r = 0.05f;

  const float v = dx * dx + dy * dy;
  if (v < r * r) {
    // out[idx(x, y, w, h)] = 1.f - v / (r * r) + 0.5f;
    out[idx(x, y, w, h)] = draw_val;
  }

  return;
}

__host__ __device__ int idx(int x, int y, int w, int h) {
  return mod(y, h) * w + mod(x, w);
}

__host__ __device__ float sigmoid_a(float x, float a, float b) {
  return 1.0 / (1.0 + exp(-(x - a) * 4.0 / b));
}

__host__ __device__ float sigmoid_b(float x, float b, float eb) {
  return 1.0 - sigmoid_a(x, b, eb);
}

__host__ __device__ float sigmoid_ab(float x, float a, float b, float ea, float eb) {
  return sigmoid_a(x, a, ea) * sigmoid_b(x, b, eb);
}

__host__ __device__ float sigmoid_mix(float x, float y, float m, float em) {
  return x * (1.0 - sigmoid_a(m, 0.5, em)) + y * sigmoid_a(m, 0.5, em);
}

// From https://www.shadertoy.com/view/XtdSDn#
__host__ __device__ float trans_shadertoy(Config& cfg, float inner, float outer) {
  // n - outer
  // m - inner
  const float alpha_n = cfg.alpha_outer;
  const float alpha_m = cfg.alpha_inner;

  const float n = outer;
  const float m = inner;

  return sigmoid_mix(sigmoid_ab(n, cfg.b1, cfg.b2, alpha_n, alpha_n),
                     sigmoid_ab(n, cfg.d1, cfg.d2, alpha_n, alpha_n), m, alpha_m);
}

__host__ __device__ float sigmoid1(float x, float a, float alpha) {
  return 1.0 / (1.0 + exp(-(x - a) * 4.0 / alpha));
}

__host__ __device__ float sigmoid2(float x, float a, float b, float alpha) {
  return sigmoid1(x, a, alpha) * (1. - sigmoid1(x, b, alpha));
}

__host__ __device__ float sigmoidm(float x, float y, float m, float alpha) {
  return x * (1. - sigmoid1(m, .5f, alpha)) + y * sigmoid1(m, .5f, alpha);
}

// From Smooth Life paper
__host__ __device__ float trans_paper(Config& cfg, float inner, float outer) {
  // n - outer
  // m - inner
  const float s1 = sigmoidm(cfg.b1, cfg.d1, inner, cfg.alpha_inner);
  const float s2 = sigmoidm(cfg.b2, cfg.d2, inner, cfg.alpha_inner);

  return sigmoid2(outer, s1, s2, cfg.alpha_outer);
}

__device__ float saturate(float v) {
  if (v < 0.f) {
    return 0.f;
  }
  if (v > 1.f) {
    return 1.f;
  }
  return v;
}

__global__ void cuda_frame(Config& cfg, float* in, float* out, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const float r_outer = 10.f;
  const float r_inner = 3.f;

  const float area_inner = PI * r_inner * r_inner;
  const float area_outer = PI * r_outer * r_outer - area_inner; // ring

  // const float aliasing_width = 1.f;

  float inner_val = 0.f;
  float outer_val = 0.f;

  for (float offset_x = -r_outer; offset_x <= r_outer; offset_x += 1.f) {
    for (float offset_y = -r_outer; offset_y <= r_outer; offset_y += 1.f) {
      // TODO: try comparing squared values
      const float r = sqrt(offset_x * offset_x + offset_y * offset_y);

      const float cx = x + offset_x;
      const float cy = y + offset_y;

      const float val = in[idx(cx, cy, w, h)];

      if (cfg.is_paper) {
        const float aliasing_width = 1.f;
        if (0) {
        } else if (r < r_inner - aliasing_width * .5f) {
          inner_val += val;
        } else if (r < r_inner + aliasing_width * .5f) {
          inner_val += val * (r_inner + aliasing_width * .5f - r) / aliasing_width;
        } else if (r < r_outer - aliasing_width * .5f) {
          outer_val += val;
        } else if (r < r_outer + aliasing_width * .5f) {
          outer_val += val * (r_outer + aliasing_width * .5f - r) / aliasing_width;
        }
      } else {
        // Shadertoy
        outer_val += val * saturate(r - r_inner + .5) * (1.f - saturate(r - r_outer + .5f));
        inner_val += val * (1.f - saturate(r - r_inner + .5f));
      }
    }
  }

  inner_val /= area_inner;
  outer_val /= area_outer;

  const float dt = .30f;
  const float prev = in[idx(x, y, w, h)];

  const float factor = cfg.is_paper ? trans_paper(cfg, inner_val, outer_val) : trans_shadertoy(cfg, inner_val, outer_val);
  float res = prev + cfg.dt * (factor * 2.f - 1.f);

  if (res < 0.f) {
    res = 0.f;
  }

  if (res > 1.f) {
    res = 1.f;
  }

  out[idx(x, y, w, h)] = res;
}

__global__ void cuda_val_to_col(float* in, unsigned char* out, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int i = y * w + x;
  const int res_i = (y * w + x) * 4;

  const int res = in[i] * 255;

  out[res_i + 0] = res;
  out[res_i + 1] = res;
  out[res_i + 2] = res;
  out[res_i + 3] = 255;
}

bool Cellular::init(unsigned char* gpu_texture_, int w_, int h_) {
  gpu_texture = gpu_texture_;
  w = w_;
  h = h_;

  blocks = dim3{w / threads.x, h / threads.y};

  cuda_buf[0] = GpuSlice<float>::alloc(w * h);
  cuda_buf[1] = GpuSlice<float>::alloc(w * h);
  cpu_buf = CpuSlice<float>::alloc(w * h);

  assert(cpu_buf.ptr);
  assert(cuda_buf[0].ptr);
  assert(cuda_buf[1].ptr);

  cpu_init(cpu_buf.ptr, w, h);
  cuda_buf[1].copy_from(cpu_buf);

  CUDA_KERNEL(cuda_init<<<blocks, threads>>>(cuda_buf[1].ptr, w, h));

  CUDA_CALL(cudaMalloc(&config_gpu, sizeof(Config)));

  return true;
}

bool Cellular::term() {
  return true;
}

bool Cellular::update() {

  if (show_demo_window) {
    ImGui::ShowDemoWindow(&show_demo_window);
  }

  {
    ImGui::Begin("Settings");

    const auto& io = ImGui::GetIO();
    ImGui::Text("%.1f FPS (%.3f ms/frame)", io.Framerate, 1000.0f / io.Framerate);

    ImGui::Checkbox("ImGui Demo Window", &show_demo_window);

    if (ImGui::RadioButton("Paper Algo", config.is_paper)) {
      config.is_paper = true;
    }
    if (ImGui::RadioButton("Shadertoy Algo", !config.is_paper)) {
      config.is_paper = false;
    }

    ImGui::SliderFloat("dt", &config.dt, 0.0f, 1.0f);
    ImGui::DragFloatRange2("birth", &config.b1, &config.b2, .001f, 0.f, 1.f);
    ImGui::DragFloatRange2("death", &config.d1, &config.d2, .001f, 0.f, 1.f);
    ImGui::SliderFloat("alpha_outer", &config.alpha_outer, 0.f, 1.f);
    ImGui::SliderFloat("alpha_inner", &config.alpha_inner, 0.f, 1.f);
    ImGui::SliderFloat("draw_value", &config.draw_value, 0.f, 1.f);

    if (ImGui::Button("Reset")) {
      cuda_buf[1].copy_from(cpu_buf);
    }

    ImGui::End();
  }

  copy_to_gpu(config_gpu, &config);

  std::swap(cuda_buf[0], cuda_buf[1]);
  CUDA_KERNEL(cuda_frame<<<blocks, threads>>>(*config_gpu, cuda_buf[0].ptr, cuda_buf[1].ptr, w, h));
  CUDA_KERNEL(cuda_val_to_col<<<blocks, threads>>>(cuda_buf[1].ptr, gpu_texture, w, h));

  return true;
}

bool Cellular::draw(int x, int y) {
  CUDA_KERNEL(cuda_draw<<<blocks, threads>>>(cuda_buf[1].ptr, w, h, x, y, config.draw_value));
  return true;
}