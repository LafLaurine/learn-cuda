#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

// __global__ indicates a function or "kernel" that runs on the device and is called from host code

__global__ void hello_kernel(void)
{
    // greet from the device : the GPU and its memory
    printf("Hello, world from the device!\n");
}

int main(void)
{
    // greet from the host : the CPU and its memory
    printf("Hello, world from the host!\n");
    
    // triple angle brackets mark a call from host code to device code
    // launch a kernel with a single thread to greet from the device
    hello_kernel<<<1,1>>>();
    
    // wait for the device to finish so that we see the message
    cudaDeviceSynchronize();

    // check error 
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

    return 0;
}

// nvcc separates source code into host and device components*
// nvcc hello.cu -o hello