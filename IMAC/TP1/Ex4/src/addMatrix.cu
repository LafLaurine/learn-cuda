#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include<time.h>

#include "chronoGPU.hpp"
#include "chronoCPU.hpp"
#include "common.hpp"

#define WIDTH 128
#define HEIGHT 128
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
    std::cout << "Output of the sum : ";

    for (int ix = 0; ix < WIDTH; ix++) { 
        for (int iy = 0; iy < HEIGHT; iy++) {
            s[iy*WIDTH + ix] = m1[iy*WIDTH + ix] + m2[iy*WIDTH + ix];
            std::cout << s[iy*WIDTH + ix] << " ";
        }
    }
}

__global__ void matSum(int* dev_a, int* dev_b, int* dev_res){
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

void GPUCompute(const int* a, const int* b, int* res) {
    ChronoGPU chrGPU;

    int* dev_a = NULL;
    int* dev_b = NULL;
    int* dev_res = NULL;

    // Allocate arrays on device (input and ouput)
    const size_t bytes = SIZE*sizeof(int);

    cudaMalloc((void**)&dev_a, bytes);
    cudaMalloc((void**)&dev_b, bytes);
    cudaMalloc((void**)&dev_res, bytes);

    //The error lays here
    cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);
        
    dim3 threadsPerBlock(WIDTH,HEIGHT);
    dim3 numBlocks(ceil((float)WIDTH / threadsPerBlock.x), ceil((float)HEIGHT/threadsPerBlock.y));
    chrGPU.start();
    matSum<<<numBlocks, threadsPerBlock>>>(dev_a,dev_b,dev_res);
    chrGPU.stop();
    std::cout 	<< "\n -> GPU Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
    cudaDeviceSynchronize();

    cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(dev_res);
    cudaFree(dev_b);
    cudaFree(dev_a);

}

int main()
{   
    ChronoCPU chrCPU;
    srand(static_cast<unsigned>(time(0)));
    int m1[SIZE];
    createMatrix(m1);
    int m2[SIZE];
    createMatrix(m2);
    int mres[SIZE] = {0};
    chrCPU.start();
    CPUCompute(m1,m2);
    chrCPU.stop();
    std::cout 	<< "\n -> CPU Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
    GPUCompute(m1,m2,mres);
	return 0;
}

