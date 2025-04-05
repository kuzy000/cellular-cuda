
main: glad main.cu
	nvcc -std=c++20 -g -G -I thirdparty/ -I glad/include -lGL -lglfw -o main main.cu glad/src/gl.c


glad: glad
	glad --api gl:core --out-path glad c
