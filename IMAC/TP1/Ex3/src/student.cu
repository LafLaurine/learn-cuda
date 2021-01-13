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
		uint x = (blockIdx.x * blockDim.x) + threadIdx.x;
		uint y = (blockIdx.y * blockDim.y) + threadIdx.y;
		uint idx = y * width + x;
		if (idx < width * height * 3)
		{
			const unsigned char r = dev_input[idx];
			const unsigned char v = dev_input[idx + 1];
			const unsigned char b = dev_input[idx + 2];
			dev_output[idx] = min(255.f, r * 0.393 + v * 0.769 + b * 0.189);
			dev_output[idx + 1] = min(255.f, r * 0.349 + v * 0.686 + b * 0.168);
			dev_output[idx + 2] = min(255.f, r * 0.272 + v * 0.534 + b * 0.131);
		}
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
		int blocks = (width * height) / 512;
		sepiaImageCUDA<<<blocks, 512>>>(width, height, dev_input, dev_output);

		// Copy data from device to host
		cudaMemcpy(output.data(), dev_output, bytes, cudaMemcpyDeviceToHost);

		// Free memory
		cudaFree(dev_input);
		cudaFree(dev_output);
	}
}
