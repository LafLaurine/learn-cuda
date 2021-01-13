/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < n) {
			dev_res[idx] = dev_a[idx] + dev_b[idx];
		}
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		int thr_per_blk = 256;
		int blk_in_grid = ceil(float(size) / thr_per_blk);

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		// allocate GPU buffers for three vectors (two input, one output)
		cudaMalloc((void**)&dev_res, size * sizeof(int));
		cudaMalloc((void**)&dev_a, size * sizeof(int));
		cudaMalloc((void**)&dev_b, size * sizeof(int));
		
		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

		// Launch kernel
		//sumArraysCUDA<<<1,256>>>(size,dev_a,dev_b,dev_res);
		//number of blocks and number of threads in a block
		sumArraysCUDA<<<blk_in_grid, thr_per_blk>>>(size,dev_a,dev_b,dev_res);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaDeviceSynchronize();

		// Copy data from device to host (output array)
		cudaMemcpy(res, dev_res, size * sizeof(int), cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
	}
}

