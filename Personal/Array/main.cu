#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// add() will execute on the device and will be called from the host
// as add runs on the device, we need to use pointers because a,b and c must point to device memory and we need to allocate memory on the GPU
__global__ void add(int *a, int *b, int *c)
{
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(void)
{
    // host copies of a, b, c
    int a[] = { 1, 2, 3 };
    int b[] = { 4, 5, 6 };
    int c[] = { 0, 0, 0 }; 

    // device copies of a, b, c
    int *d_a, *d_b, *d_c; 
    int size = sizeof(int) * 3;

    // we need to allocate memory on the GPU
    // allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // launch add() kernel on the GPU with a single thread
    add<<<1,3>>>(d_a, d_b, d_c);

    // copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // don't forget to free the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (size_t i = 0; i < 3; i++)
    {
        printf("Val is : %d \n", c[i]);
    }

    // check error 
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

    return 0;
}
