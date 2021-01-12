#include <stdio.h>
#define N (2048*2048)
#define THREADS_PER_BLOCK 512

void random_ints(int* a, int n)
{
   int i;
   for (i = 0; i < n; ++i) {
       a[i] = rand() %5000;
   }
}

// indexing an array with one element per thread
// M threads per block, a unique index for each thread is given by threadIdx.x + blockIdx.x * M

__global__ void add(int *a, int *b, int *c, int n)
{
    //blockDim.x represents threads per block
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // as we need to avoid to go beyond the end of the arrays, we need to define the limit
    if (index < n)
        c[index] = a[index] + b[index];
}

int main(void)
{
    // host copies of a, b, c
    int *a, *b, *c; 
    // device copies of a, b, c
    int *d_a, *d_b, *d_c; 
    int size = N * sizeof(int);

    // we need to allocate memory on the GPU
    // allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // allocate space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); 
    random_ints(a, N);
    b = (int *)malloc(size); 
    random_ints(b, N);
    c = (int *)malloc(size);

    // copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // launch add() kernel on the GPU
    add<<<(N + THREADS_PER_BLOCK-1)/ THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // don't forget to free the memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // check error 
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

    return 0;
}