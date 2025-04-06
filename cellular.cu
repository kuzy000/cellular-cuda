
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
__host__ __device__ float trans_shadertoy(float inner, float outer) {
  // n - outer
  // m - inner
  const float b1 = 0.257f;
  const float b2 = 0.336f;
  const float d1 = 0.365f;
  const float d2 = 0.549f;
  const float alpha_outer = 0.028f;
  const float alpha_inner = 0.147f;
  const float alpha_n = alpha_outer;
  const float alpha_m = alpha_inner;

  const float n = outer;
  const float m = inner;

  return sigmoid_mix(sigmoid_ab(n, b1, b2, alpha_n, alpha_n),
                     sigmoid_ab(n, d1, d2, alpha_n, alpha_n), m, alpha_m);
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
__host__ __device__ float trans_paper(float inner, float outer) {
  // n - outer
  // m - inner
  const float b1 = 0.278f;
  const float b2 = 0.365f;
  const float d1 = 0.267f;
  const float d2 = 0.445f;
  const float alpha_outer = 0.028f;
  const float alpha_inner = 0.147f;
  const float s1 = sigmoidm(b1, d1, inner, alpha_inner);
  const float s2 = sigmoidm(b2, d2, inner, alpha_inner);

  return sigmoid2(outer, s1, s2, alpha_outer);
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

__global__ void cuda_frame(float* in, float* out, int w, int h) {
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

      // Smoothlife paper
      // if (0) {
      //} else if (r < r_inner - aliasing_width * .5f) {
      //    inner_val += val;
      //} else if (r < r_inner + aliasing_width * .5f) {
      //    inner_val += val * (r_inner + aliasing_width * .5f - r) / aliasing_width;
      //} else if (r < r_outer - aliasing_width * .5f) {
      //    outer_val += val;
      //} else if (r < r_outer + aliasing_width * .5f) {
      //    outer_val += val * (r_outer + aliasing_width * .5f - r) / aliasing_width;
      //}

      // Shadertoy
      outer_val += val * saturate(r - r_inner + .5) * (1.f - saturate(r - r_outer + .5f));
      inner_val += val * (1.f - saturate(r - r_inner + .5f));
    }
  }

  inner_val /= area_inner;
  outer_val /= area_outer;

  const float dt = .30f;
  const float prev = in[idx(x, y, w, h)];
  float res = prev + dt * (trans_shadertoy(inner_val, outer_val) * 2.f - 1.f);

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