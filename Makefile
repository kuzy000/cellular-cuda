.PHONY: main

override CFLAGS += -std=c++20
override CFLAGS += -arch=native
override CFLAGS += -Xcompiler -fno-exceptions,-static-libgcc,-static-libstdc++
override CFLAGS += -lineinfo -g -Xptxas -O3

override CFLAGS += -I build/glad/include
override CFLAGS += -I thirdparty/glm/include
override CFLAGS += -I thirdparty/glm/include
override CFLAGS += -I thirdparty/glm/include
override CFLAGS += -I thirdparty/imgui
override CFLAGS += -I thirdparty/imgui/backends

override CFLAGS += -lGL -lglfw

IMGUI_OBJ += build/imgui/imgui.o
IMGUI_OBJ += build/imgui/imgui_demo.o
IMGUI_OBJ += build/imgui/imgui_draw.o
IMGUI_OBJ += build/imgui/imgui_tables.o
IMGUI_OBJ += build/imgui/imgui_widgets.o
IMGUI_OBJ += build/imgui/imgui_impl_glfw.o
IMGUI_OBJ += build/imgui/imgui_impl_opengl3.o

$(shell mkdir -p build/imgui)

main: build/glad/src/gl.c $(IMGUI_OBJ)
	nvcc $(CFLAGS) -o $@ src/main.cu $^

build/imgui/%.o: thirdparty/imgui/%.cpp
	nvcc $(CFLAGS) -c -o $@ $<

build/imgui/%.o: thirdparty/imgui/backends/%.cpp
	nvcc $(CFLAGS) -c -o $@ $<

build/glad/src/gl.c:
	glad --api gl:core --out-path build/glad c
