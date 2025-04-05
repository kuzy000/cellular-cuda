#include <cstddef>
#include <cstdio>
#include <cstring>
#include <glad/gl.h>
#include <utility>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <stdlib.h>
#include <stdio.h>
#include <cassert>

#include <chrono>
#include <thread>

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

__host__ __device__ bool get_cell(unsigned char* buf, int x, int y, int w, int h) {
    const int i = (y * w + x) * 4;
    return buf[i] > 0;
}

__host__ __device__ void set_cell(unsigned char* buf, int x, int y, int w, int h, bool val) {
    const int i = (y * w + x) * 4;
    const int v = val ? 255 : 0;
    buf[i + 0] = v;
    buf[i + 1] = v;
    buf[i + 2] = v;
    buf[i + 3] = v;
}

__host__ __device__ int mod(int a, int b) {
    const int r = a % b;
    return r >= 0 ? r : r + b;
}

void cpu_init(unsigned char* out, int w, int h) {
    memset(out, 0, w * h * 4);
    
    const char* data = R"game(
    

.........................x
.......................x.x
.............xx......xx............xx
............x...x....xx............xx
.xx........x.....x...xx
.xx........x...x.xx....x.x
...........x.....x.......x
............x...x
.............xx..
)game";
    
    const int start_x = 20;
    int x = start_x;
    int y = 100;
    
    while (*data != '\0') {
        if (*data == 'x') {
            set_cell(out, x, y, w, h, true);
        }
        
        if (*data == '\n') {
            y -= 1; 
            x = start_x;
        }
        
        x += 1;
        data += 1;
    }
}

__global__ void cuda_init(unsigned char* out, int w, int h)
{
    return;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int i = (y * w + x) * 4;

    float t0 = (float)x / w;
    float t1 = (float)y / h;

    out[i + 0] = t0 * 255;
    out[i + 1] = 0;
    out[i + 2] = t1 * 255;
    out[i + 3] = 255;
}

__global__ void cuda_frame(unsigned char* in, unsigned char* out, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int n = 0;
    n += get_cell(in, mod(x - 1, w), mod(y - 1, h), w, h);
    n += get_cell(in, mod(x - 0, w), mod(y - 1, h), w, h);
    n += get_cell(in, mod(x + 1, w), mod(y - 1, h), w, h);

    n += get_cell(in, mod(x - 1, w), mod(y - 0, h), w, h);
    //n += get_cell(in, mod(x - 0, w), mod(y - 0, h), w, h);
    n += get_cell(in, mod(x + 1, w), mod(y - 0, h), w, h);
    
    n += get_cell(in, mod(x - 1, w), mod(y + 1, h), w, h);
    n += get_cell(in, mod(x - 0, w), mod(y + 1, h), w, h);
    n += get_cell(in, mod(x + 1, w), mod(y + 1, h), w, h);
    
    if (get_cell(in, x, y, w, h)) {
        set_cell(out, x, y, w, h, n >= 2 && n <= 3);
    }
    else {
        set_cell(out, x, y, w, h, n == 3);
    }
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

    const int size_bytes = 128 * 128 * 4;
    
    unsigned char* cuda_buf[2];
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
    
    const dim3 image_size{128, 128};
    
    const dim3 threads{16, 16};
    const dim3 blocks{
        image_size.x / threads.x,
        image_size.y / threads.y,
    };


    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 128, 128, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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
    
    unsigned char* cpu_buf = (unsigned char*)malloc(size_bytes);
    cpu_init(cpu_buf, 128, 128);

    res = cudaMemcpy(cuda_buf[1], cpu_buf, size_bytes, cudaMemcpyHostToDevice);
    if (res != cudaSuccess) {
      printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
      exit(1);
    }

    cuda_init<<<blocks, threads>>>(cuda_buf[1], 128, 128);

    res = cudaPeekAtLastError();
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }
    
    const float fixed_fps = 60.f;
    
    int frame_number = 0;
    
    while (!glfwWindowShouldClose(window)) {
        const auto beg_time = std::chrono::steady_clock::now();

        if (frame_number % 5 == 0) {
          std::swap(cuda_buf[0], cuda_buf[1]);
          cuda_frame<<<blocks, threads>>>(cuda_buf[0], cuda_buf[1], 128, 128);

          res = cudaPeekAtLastError();
          if (res != cudaSuccess) {
            printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
            exit(1);
          }

          res = cudaMemcpyToArray(cuda_arr, 0, 0, cuda_buf[1], size_bytes, cudaMemcpyDeviceToDevice);
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
        
        const auto sleep_dur = std::chrono::duration<float>{1.f / fixed_fps} - dt; std::this_thread::sleep_for(sleep_dur);
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

