/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sepiaImageCUDA(const int width, const int height, const unsigned char* dev_input, unsigned char* dev_output)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		dev_output[idx] = dev_input[idx];
	}

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;

		// Allocate memory
		const size_t bytes = width * height * 3 * sizeof(uchar);
		cudaMalloc(&dev_input, bytes);
		cudaMalloc(&dev_output, bytes);
		
		// Copy data from host to device
		cudaMemcpy(dev_input, input.data(), bytes, cudaMemcpyHostToDevice);

		// Launch kernel
		sepiaImageCUDA<<<1, 256>>>(width, height, dev_input, dev_output);

		// Copy data from device to host
		cudaMemcpy(output.data(), dev_output, bytes, cudaMemcpyDeviceToHost);

		// Free memory
		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}
