.PHONY: main

override CFLAGS += -std=c++20
override CFLAGS += -arch=native
override CFLAGS += -Xcompiler -fno-exceptions,-static-libgcc,-static-libstdc++
override CFLAGS += -lineinfo -g -Xptxas -O3

override CFLAGS += -I thirdparty/glad/include
override CFLAGS += -I thirdparty/glm/include

override CFLAGS += -lGL -lglfw

main: thirdparty/glad
	nvcc $(CFLAGS) -o main src/main.cu thirdparty/glad/src/gl.c

thirdparty/glad:
	glad --api gl:core --out-path thirdparty/glad c
