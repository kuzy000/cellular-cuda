#include "app.cuh"

#include "common.cuh"

#include <GLFW/glfw3.h>
#include <stdio.h>
#include <unistd.h>

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

bool App::init() {
  glfwSetErrorCallback([](int error, const char* desc) {
    printf("ERROR: %s (%d)\n", desc, error);
  });

  if (!glfwInit()) {
    printf("ERROR: Failed to init GLFW");
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  window = glfwCreateWindow(640, 480, "Cellular CUDA", nullptr, nullptr);
  if (!window) {
    printf("ERROR: Failed to create GLFW window");
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window);
  const int gl_ver = gladLoadGL(glfwGetProcAddress);
  printf("INFO: GL version: %d.%d\n", GLAD_VERSION_MAJOR(gl_ver),
         GLAD_VERSION_MINOR(gl_ver));
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  if (!ImGui_ImplOpenGL3_Init("#version 130")) {
    printf("ERROR: Failed to ImGui_ImplOpenGL3_Init\n");
    return false;
  }

  glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
  });

  static const struct Vertex {
    float x, y;
    float u, v;
  } vertices[6] = {
      {-1.f, -1.f, 0.f, 0.f},
      {1.f, -1.f, 1.f, 0.f},
      {1.f, 1.f, 1.f, 1.f},
      {-1.f, -1.f, 0.f, 0.f},
      {1.f, 1.f, 1.f, 1.f},
      {-1.f, 1.f, 0.f, 1.f},
  };

  glGenBuffers(1, &buf_vert);
  glBindBuffer(GL_ARRAY_BUFFER, buf_vert);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  {
    shader_vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(shader_vert, 1, &vertex_shader_text, NULL);
    glCompileShader(shader_vert);

    int status;
    glGetShaderiv(shader_vert, GL_COMPILE_STATUS, &status);
    if (!status) {
      char err[512];
      glGetShaderInfoLog(shader_vert, 512, nullptr, err);
      printf("ERROR: Vertex shader compilation error:\n%s\n", err);
      return false;
    }
  }

  {
    shader_frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shader_frag, 1, &fragment_shader_text, NULL);
    glCompileShader(shader_frag);

    int status;
    glGetShaderiv(shader_frag, GL_COMPILE_STATUS, &status);
    if (!status) {
      char err[512];
      glGetShaderInfoLog(shader_frag, 512, nullptr, err);
      printf("ERROR: Fragment shader compilation error:\n%s\n", err);
      return false;
    }
  }

  shader_prog = glCreateProgram();
  glAttachShader(shader_prog, shader_vert);
  glAttachShader(shader_prog, shader_frag);
  glLinkProgram(shader_prog);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (const void*)offsetof(Vertex, x));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (const void*)offsetof(Vertex, u));

  CUDA_CALL(cudaMalloc(&cuda_tex, w * h * 4));

  blocks = dim3{w / threads.x, h / threads.y};

  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  CUDA_CALL(cudaGraphicsGLRegisterImage(&tex_res, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

  CUDA_CALL(cudaGraphicsMapResources(1, &tex_res, 0));

  CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&cuda_arr, tex_res, 0, 0));

  cellular.init(cuda_tex, w, h);

  return true;
}

bool App::term() {
  cellular.term();

  CUDA_CALL(cudaGraphicsUnmapResources(1, &tex_res, 0));

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return true;
}

bool App::loop() {
  ImGuiIO& io = ImGui::GetIO();

  while (!glfwWindowShouldClose(window)) {
    const double beg_time = glfwGetTime();

    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    cellular.update();

    CUDA_CALL(cudaMemcpyToArray(cuda_arr, 0, 0, cuda_tex, w * h * 4, cudaMemcpyDeviceToDevice));
    ImGui::Render();

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glUseProgram(shader_prog);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);

    const double end_time = glfwGetTime();
    const double dt = end_time - beg_time;

    usleep(dt * 1e6);
  }

  return true;
}