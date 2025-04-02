#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <stdlib.h>
#include <stdio.h>
#include <cassert>

static const struct
{
    float x, y;
    float r, g, b;
} vertices[3] =
{
    { -0.6f, -0.4f, 1.f, 0.f, 0.f },
    {  0.6f, -0.4f, 0.f, 1.f, 0.f },
    {   0.f,  0.6f, 0.f, 0.f, 1.f }
};

static const char* vertex_shader_text =
"#version 330 core\n"
"in vec3 vCol;\n"
"in vec2 vPos;\n"
"out vec3 color;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(vPos, 0.0, 1.0);\n"
"    color = vCol;\n"
"}\n";

static const char* fragment_shader_text =
"#version 330 core\n"
"in vec3 color;\n"
"uniform sampler2D tex;"
"void main()\n"
"{\n"
"    vec4 c = texture(tex, gl_FragCoord.xy);"
"    gl_FragColor = vec4(c.xyz, 1.0);\n"
"}\n";

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

__global__ void cudaFill(unsigned char* col)
{
    const int i = (blockIdx.x * 128 + threadIdx.x) * 3;
    col[i + 0] = 255;
    col[i + 1] = 0;
    col[i + 2] = 0;
}

int main(void)
{
    GLFWwindow* window;
    GLuint vertex_buffer, vertex_shader, fragment_shader, program;
    GLint vpos_location, vcol_location;

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

    vpos_location = glGetAttribLocation(program, "vPos");
    vcol_location = glGetAttribLocation(program, "vCol");

    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
                          sizeof(vertices[0]), (void*) 0);
    glEnableVertexAttribArray(vcol_location);
    glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE,
                          sizeof(vertices[0]), (void*) (sizeof(float) * 2));



    cudaError_t res;



    const int size_bytes = 128 * 128 * 3;

    unsigned char* cuda_buf;
    res = cudaMalloc(&cuda_buf, size_bytes);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }


    cudaFill<<<128, 128>>>(cuda_buf);

    res = cudaPeekAtLastError();
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }

    unsigned char* cpu_buf = (unsigned char*)malloc(size_bytes);
    res = cudaMemcpy(cpu_buf, cuda_buf, size_bytes, cudaMemcpyDeviceToHost);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }

    for (int i = 0; i < size_bytes; ++i) {
        // assert(cpu_buf[i] == 128);
    }



    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

//    auto* data = new float[128*128*3];
//    for (int i = 0; i < 128*128*3; ++i) {
//        data[i] = 1.f;
//    }
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, 128, 128, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, nullptr);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // glBindTexture(GL_TEXTURE_2D, 0);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 128, 128, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


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

    res = cudaMemcpyToArray(cuda_arr, 0, 0, cuda_buf, size_bytes, cudaMemcpyDeviceToDevice);
    if (res != cudaSuccess) {
        printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
        exit(1);
    }

     res = cudaGraphicsUnmapResources(1, &tex_res, 0);
     if (res != cudaSuccess) {
         printf("ERROR: %s: %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
         exit(1);
     }

    while (!glfwWindowShouldClose(window))
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);

        glUseProgram(program);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

