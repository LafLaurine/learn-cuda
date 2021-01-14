/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "student.hpp"
#include "chronoGPU.hpp"
#include <algorithm>

#define BLOCK_SIZE 32

namespace IMAC
{
	__global__ void applyFilter(const unsigned char* dev_input, const uint width, const uint height, unsigned char* dev_output)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int index = (idy * width + idx) * 3;
		if (idx < width && idy < height){
			//replace pixel
			dev_output[index] = min(255.0f, (dev_input[index] * 0.393f + dev_input[index+1] * 0.769f + dev_input[index+2] * 0.189f));
			dev_output[index+1] = min(255.0f, (dev_input[index] * 0.349f + dev_input[index+1] * 0.686f + dev_input[index+2] * 0.168f));
			dev_output[index+2] = min(255.0f, (dev_input[index] * 0.272f + dev_input[index+1] * 0.534f + dev_input[index+2] * 0.131f));
		}
	} 

	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;

		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 numBlocks(ceil((float)width / threadsPerBlock.x), ceil((float)height/threadsPerBlock.y));
		
		chrGPU.start();

		// allocate GPU buffers
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, width * height * 3 * sizeof(uchar)));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, width * height * 3 * sizeof(uchar)));

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		
		// Copy data from host to device (input arrays) 
		HANDLE_ERROR(cudaMemcpy(dev_input, input.data(), (width * height * 3) * sizeof(uchar), cudaMemcpyHostToDevice));

		//launch kernel
		applyFilter<<<numBlocks,threadsPerBlock>>>(dev_input,width,height,dev_output);

		//HANDLE_ERROR(cudaDeviceSynchronize());

 		// Copy data from device to host (output array)
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, (width * height * 3) * sizeof(uchar), cudaMemcpyDeviceToHost));

		// Free arrays on device
		cudaFree(dev_output);
		cudaFree(dev_input);
	}
}
