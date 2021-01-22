/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

		int thr_per_blk = 512;
		int blk_in_grid = ceil(float(size) / thr_per_blk);

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		
		// allocate GPU buffers for three vectors (two input, one output)
		cudaMalloc((void**)&dev_res, bytes);
		cudaMalloc((void**)&dev_a, bytes);
		cudaMalloc((void**)&dev_b, bytes);
		

		// Copy data from host to device (input arrays) 
		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

		// Launch kernel
		//sumArraysCUDA<<<1,256>>>(size,dev_a,dev_b,dev_res);
		//number of blocks and number of threads in a block
		chrGPU.start();
		sumArraysCUDA<<<blk_in_grid, thr_per_blk>>>(size,dev_a,dev_b,dev_res);
		chrGPU.stop();

		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaDeviceSynchronize();

		// Copy data from device to host (output array)
		cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(dev_res);
		cudaFree(dev_b);
		cudaFree(dev_a);
	}
}

