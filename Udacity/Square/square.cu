#include <stdio.h>
#include <iostream>

__global__ void square(float* d_out, float* d_in) 
{
    int idx = threadIdx.x;
    d_out[idx] = d_in[idx] * d_in[idx];
}

int main() 
{
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    //generate input array on host
    float h_in[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    //declare GPU memory pointers
    float* d_in;
    float* d_out;

    // allocate GPU memory
    cudaMalloc((void**) &d_in, ARRAY_BYTES);
    cudaMalloc((void**) &d_out, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    square<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // copy back the result array to the CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    
    //print out the resulting array
    for(int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << h_out[i] << std::endl;
    }

    //we don't forget to free the GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}