#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "chronoGPU.hpp"
#include "common.hpp"

#define WIDTH 2
#define HEIGHT 2
#define SIZE HEIGHT*WIDTH

void createMatrix(int m[SIZE])
{
    for (int i = 0; i < SIZE; i++) {
        m[i] = rand()%100;
    }
}

void matricesSum(const int m1[SIZE], const int m2[SIZE])
{   
    int s[SIZE] = {0};
    std::cout << "Output of the sum : " << std::endl;

    for (int ix = 0; ix < WIDTH; ix++) { 
        for (int iy = 0; iy < HEIGHT; iy++) {
            s[iy*WIDTH + ix] = m1[iy*WIDTH + ix] + m2[iy*WIDTH + ix];
            std::cout << s[iy*WIDTH + ix] << std::endl;
        }
    }
}

__global__ void matSum(const int dev_a[SIZE], const int dev_b[SIZE], int dev_res[SIZE]){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (idy * WIDTH + idx);
    if (idx < WIDTH && idy < HEIGHT){
        dev_res[index] = dev_a[index] + dev_b[index];
    }
}

void CPUCompute(const int m1[SIZE], const int m2[SIZE]) {
    matricesSum(m1,m2);
}

void GPUCompute(const int a[SIZE], const int b[SIZE], int res[SIZE]) {
    const int* dev_a[SIZE];
    const int* dev_b[SIZE];
    int* dev_res[SIZE];

    // Allocate arrays on device (input and ouput)
    const size_t bytes = SIZE*sizeof(int);

    cudaMalloc((void**)&dev_a, bytes);
    cudaMalloc((void**)&dev_b, bytes);
    cudaMalloc((void**)&dev_res, bytes);

    //The error lays here
    cudaMemcpy(dev_a, &a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, bytes, cudaMemcpyHostToDevice);
        
    int numBlocks = 1;
    dim3 threadsPerBlock(WIDTH*HEIGHT,HEIGHT*WIDTH);
    matSum<<<numBlocks, threadsPerBlock>>>(*dev_a,*dev_b,*dev_res);
    cudaDeviceSynchronize();

    cudaMemcpy(&res, dev_res, bytes, cudaMemcpyDeviceToHost);

    std::cout << *res << std::endl;

    cudaFree(dev_res);
    cudaFree(dev_b);
    cudaFree(dev_a);

}

int main()
{   
    int m1[SIZE];
    createMatrix(m1);
    int m2[SIZE];
    createMatrix(m2);
    int mres[SIZE];

    CPUCompute(m1,m2);
    GPUCompute(m1,m2,mres);
	return 0;
}

