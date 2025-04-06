#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <glad/gl.h>
#include <utility>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <curand.h>

#include <stdlib.h>
#include <stdio.h>
#include <cassert>

#include <chrono>
#include <thread>


constexpr float PI = 3.141592654f;

static const struct Vertex
{
    float x, y;
    float u, v;
} vertices[6] =
{
    { -1.f, -1.f, 0.f, 0.f, },
    {  1.f, -1.f, 1.f, 0.f, },
    {  1.f,  1.f, 1.f, 1.f, },

    { -1.f, -1.f, 0.f, 0.f },
    {  1.f,  1.f, 1.f, 1.f },
    { -1.f,  1.f, 0.f, 1.f },
};

static const char* vertex_shader_text = R"glsl(
#version 330 core
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;

out vec2 uvOut;

void main()
{
    gl_Position = vec4(pos, 0., 1.0);
    uvOut = uv;
}
)glsl";

static const char* fragment_shader_text = R"glsl(
#version 330 core

in vec2 uvOut;

uniform sampler2D tex;

void main()
{
    vec4 c = texture(tex, uvOut.xy);
    gl_FragColor = vec4(c.xyz, 1.0);
}
)glsl";

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

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

__global__ void cuda_init(float* out, int w, int h)
{
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
                       sigmoid_ab(n, d1, d2, alpha_n, alpha_n), m, alpha_m
                      );
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
    if (v < 0.f) { return 0.f; }
    if (v > 1.f) { return 1.f; }
    return v;
}

__global__ void cuda_frame(float* in, float* out, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const float r_outer = 10.f;
    const float r_inner = 3.f;

    const float area_inner = PI * r_inner * r_inner;
    const float area_outer = PI * r_outer * r_outer - area_inner; // ring

    const float aliasing_width = 1.f;

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
            //if (0) {
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

__global__ void cuda_val_to_col(float* in, unsigned char* out, int w, int h)
{
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

int main(void)
{
    GLFWwindow* window;
    GLuint vertex_buffer, vertex_shader, fragment_shader, program;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetKeyCallback(window, key_callback);

    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);

    // NOTE: OpenGL error checks have been omitted for brevity

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    {
        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
        glCompileShader(vertex_shader);

        int status;
        glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &status);
        if (!status) {
            char err[512];
            glGetShaderInfoLog(vertex_shader, 512, nullptr, err);
            printf("%s\n", err);
        }
    }

    {
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
        glCompileShader(fragment_shader);

        int status;
        glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &status);
        if (!status) {
            char err[512];
            glGetShaderInfoLog(fragment_shader, 512, nullptr, err);
            printf("%s\n", err);
        }
    }

    program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);


    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          sizeof(vertices[0]), (const void*)offsetof(Vertex, x));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                          sizeof(vertices[0]), (const void*)offsetof(Vertex, u));


    cudaError_t res;
    
    const int w = 1024;
    const int h = 1024;

    const int size_bytes = w * h * sizeof(float);

    unsigned char* cuda_tex;
    const int tex_bytes = w * h * 4;

    float* cuda_buf[2];
    res = cudaMalloc(&cuda_buf[0], size_bytes);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }

    res = cudaMalloc(&cuda_buf[1], size_bytes);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }
    
    res = cudaMalloc(&cuda_tex, tex_bytes);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }

    const dim3 image_size{w, h};
    
    const dim3 threads{16, 16};
    const dim3 blocks{
        image_size.x / threads.x,
        image_size.y / threads.y,
    };


    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


    cudaGraphicsResource* tex_res;
    res = cudaGraphicsGLRegisterImage(&tex_res, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }

    res = cudaGraphicsMapResources(1, &tex_res, 0);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }

    cudaArray* cuda_arr;
    res = cudaGraphicsSubResourceGetMappedArray(&cuda_arr, tex_res, 0, 0);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }
    
    float* cpu_buf = (float*)malloc(size_bytes);
    cpu_init(cpu_buf, w, h);

    res = cudaMemcpy(cuda_buf[1], cpu_buf, size_bytes, cudaMemcpyHostToDevice);
    if (res != cudaSuccess) {
      printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
      exit(1);
    }

    cuda_init<<<blocks, threads>>>(cuda_buf[1], w, h);

    res = cudaPeekAtLastError();
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }
    
    const float fixed_fps = 60.f;
    
    int frame_number = 0;
    
    while (!glfwWindowShouldClose(window)) {
        const auto beg_time = std::chrono::steady_clock::now();

        if (frame_number % 1 == 0) {
          std::swap(cuda_buf[0], cuda_buf[1]);
          cuda_frame<<<blocks, threads>>>(cuda_buf[0], cuda_buf[1], w, h);

          res = cudaPeekAtLastError();
          if (res != cudaSuccess) {
            printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
            exit(1);
          }

          cuda_val_to_col<<<blocks, threads>>>(cuda_buf[1], cuda_tex, w, h);

          res = cudaMemcpyToArray(cuda_arr, 0, 0, cuda_tex, tex_bytes, cudaMemcpyDeviceToDevice);
          if (res != cudaSuccess) {
            printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
            exit(1);
          }
        }

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);

        glUseProgram(program);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
        
        const auto end_time = std::chrono::steady_clock::now();
        const auto dt = end_time - beg_time;
        
        const auto sleep_dur = std::chrono::duration<float>{1.f / fixed_fps} - dt; 
        std::this_thread::sleep_for(sleep_dur);
        frame_number += 1;
    }

    res = cudaGraphicsUnmapResources(1, &tex_res, 0);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

