#include <stdio.h>

// add() will execute on the device and will be called from the host
// as add runs on the device, we need to use pointers because a,b and c must point to device memory and we need to allocate memory on the GPU
__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
    printf("Result %d ", *c);
}

int main(void)
{
    // host copies of a, b, c
    int a, b, c; 
    // device copies of a, b, c
    int *d_a, *d_b, *d_c; 
    int size = sizeof(int);

    // we need to allocate memory on the GPU
    // allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = 5;
    b = 3;

    // copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // launch add() kernel on the GPU with a single thread
    add<<<1,1>>>(d_a, d_b, d_c);

    // copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // don't forget to free the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // check error 
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

    return 0;
}