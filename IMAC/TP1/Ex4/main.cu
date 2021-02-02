#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

void CPUMatrixSum(const int* m1, const int* m2, const int height, const int width, int* res)
{   
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            const int index = iy*width + ix;
            res[index] = m1[index] + m2[index];
        }
    }
}

__global__ void GPUMatrixSum(const int* m1, const int* m2, const int height, const int width, int* res)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (idy * width + idx);
    if (idx < width && idy < height){
        res[index] = m1[index] + m2[index];
    }
}

void GPUCompute(const int* m1, const int* m2, const int height, const int width, int* res) 
{
    int* dev_m1 = nullptr;
    int* dev_m2 = nullptr;
    int* dev_res = nullptr;

    // Allocate arrays on device (input and ouput)
    const size_t bytes = width * height *sizeof(int);
    cudaMalloc((void**)&dev_m1, bytes);
    cudaMalloc((void**)&dev_m2, bytes);
    cudaMalloc((void**)&dev_res, bytes);

    cudaMemcpy(dev_m1, m1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_m2, m2, bytes, cudaMemcpyHostToDevice);
        
    int numBlocks = 1;
    dim3 threadsPerBlock(width*height, width*height);
    GPUMatrixSum<<<numBlocks, threadsPerBlock>>>(dev_m1, dev_m2, height, width, dev_res);

    cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(dev_res);
    cudaFree(dev_m2);
    cudaFree(dev_m1);
}

void DisplayMatrix(int* m, const int height, const int width)
{
    for (int ix = 0; ix < width; ix++) { 
        for (int iy = 0; iy < height; iy++) {
            std::cout << m[iy*width + ix] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    const int width = 2;
    const int height = 2;

    int m1[width * height];
    std::fill(m1, m1 + width * height, 1);

    int m2[width * height];
    std::fill(m2, m2 + width * height, 2);

    std::cout << "CPU compute :" << std::endl;
    int mres[width * height] = {0};
    CPUMatrixSum(m1, m2, height, width, mres);
    DisplayMatrix(mres, width, height);

    std::cout << "GPU compute :" << std::endl;
    std::fill(mres, mres + width * height, 0);
    GPUCompute(m1, m2, height, width, mres);
    DisplayMatrix(mres, width, height);
	return 0;
}
