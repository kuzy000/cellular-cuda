
main: glad main.cu
	nvcc -lineinfo -O3 -Xptxas -O3 -std=c++20 -g -I thirdparty/ -I glad/include -lGL -lglfw -o main main.cu glad/src/gl.c


glad: glad
	glad --api gl:core --out-path glad c
