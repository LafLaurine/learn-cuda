/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}
// ==================================================

	__global__ void convGPU(const uint imgWidth, const uint imgHeight, uchar4* inputImg, float* matConv, uchar4* output)
	{
		printf("Hello from GPU %d \n", imgWidth);
	}

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar4* d_inputImg = nullptr;
		float *d_matConv = nullptr;
		uchar4* d_output = nullptr;

		// Allocate arrays
		cudaMalloc(&d_inputImg, sizeof(uchar4) * inputImg.size());
		cudaMalloc(&d_matConv, sizeof(float) * matConv.size());
		cudaMalloc(&d_output, sizeof(uchar4) * output.size());

		// Copy data from host to device
		cudaMemcpy(d_inputImg, inputImg.data(), sizeof(uchar4) * inputImg.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_matConv, matConv.data(), sizeof(float) * matConv.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(d_output, output.data(), sizeof(uchar4) * output.size(), cudaMemcpyHostToDevice);

		// Launch kernel
		convGPU<<<1, 1>>>(imgWidth, imgHeight, d_inputImg, d_matConv, d_output);
		cudaDeviceSynchronize();

		// Copy data from device to host (output array)  
		cudaMemcpy(output.data(), d_output, sizeof(uchar4) * output.size(), cudaMemcpyDeviceToHost);

		// Free arrays on device
		cudaFree(d_inputImg);
		cudaFree(d_matConv);
		cudaFree(d_output);
	}
}
