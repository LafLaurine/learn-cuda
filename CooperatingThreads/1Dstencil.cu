#include <stdio.h>

#define BLOCK_SIZE 256
#define RADIUS 3
#define NUM_ELEMENTS (4096*2)

// CUDA API error checking macro
#define cudaCheck(error) \
    if (error != cudaSuccess) { \
        printf("Fatal error: %s at %s:%d\n", \
            cudaGetErrorString(error), \
            __FILE__, __LINE__); \
        exit(1); \
    }

// each output element is the sum of input elements within a radius
// if radius is 3, then each output element is the sum of 7 input elements
__global__ void stencil_1d(int *in, int *out) {
    // within a block, threads share data via shared memory ("global memory")
    // data is not visible to threads in other blocks
    // use __shared__ to declare a var/array in shared memory

    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    // each thread processs one output element (blockDim.x elements per block)
    int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + RADIUS;
    int lindex = threadIdx.x + RADIUS;
    
    // read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // synchronize all threads in the block : ensure all data is available
    __syncthreads();

    // apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++) {
        result += temp[lindex + offset];
    }

    // store the result
    out[gindex-RADIUS] = result;
}

int main(void)
{
    unsigned int i;
    int h_in[NUM_ELEMENTS + 2 * RADIUS], h_out[NUM_ELEMENTS];
    int *d_in, *d_out;

    // initialize host data
    for( i = 0; i < (NUM_ELEMENTS + 2*RADIUS); ++i )
        h_in[i] = 1; // With a value of 1 and RADIUS of 3, all output values should be 7

    // allocate space on the device
    cudaCheck( cudaMalloc( &d_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int)) );
    cudaCheck( cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int)) );

    // copy input data to device
    cudaCheck( cudaMemcpy( d_in, h_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int), cudaMemcpyHostToDevice) );

    stencil_1d<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_in, d_out);

    // copy result back to host
    cudaCheck( cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost) );

    // verify every out value is 7
    for( i = 0; i < NUM_ELEMENTS; ++i ) {
        if (h_out[i] != 7)
        {
            printf("Element h_out[%d] == %d != 7\n", i, h_out[i]);
            break;
        }
    }

    if (i == NUM_ELEMENTS)
        printf("SUCCESS!\n");

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);

  return 0;
}