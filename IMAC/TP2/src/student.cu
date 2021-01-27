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

	__device__ float cu_clampf(const float val, const float minVal , const float maxVal)
	{
		return min(maxVal, max(minVal, val));
	}

	__global__ void convGPU(const uint imgWidth, const uint imgHeight, const uint matSize, uchar4* inputImg, float* matConv, uchar4* output)
	{
		const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
   		const uint idy = blockIdx.y * blockDim.y + threadIdx.y;
		const uint index = idy * imgWidth + idx;
		if (idx < imgWidth && idy < imgHeight)
		{
			float3 sum = make_float3(0.f,0.f,0.f);
			// Apply convolution
			for ( uint j = 0; j < matSize; ++j ) 
			{
				for ( uint i = 0; i < matSize; ++i ) 
				{
					int dX = idx + i - matSize / 2;
					int dY = idy + j - matSize / 2;

					// Handle borders
					if ( dX < 0 ) 
						dX = 0;

					if ( dX >= imgWidth ) 
						dX = imgWidth - 1;

					if ( dY < 0 ) 
						dY = 0;

					if ( dY >= imgHeight ) 
						dY = imgHeight - 1;

					const int idMat		= j * matSize + i;
					const int idPixel	= dY * imgWidth + dX;
					sum.x += (float)inputImg[idPixel].x * matConv[idMat];
					sum.y += (float)inputImg[idPixel].y * matConv[idMat];
					sum.z += (float)inputImg[idPixel].z * matConv[idMat];
				}
			}
			output[index].x = (uchar)cu_clampf( sum.x, 0.f, 255.f );
			output[index].y = (uchar)cu_clampf( sum.y, 0.f, 255.f );
			output[index].z = (uchar)cu_clampf( sum.z, 0.f, 255.f );
			output[index].w = 255;
		}
	}

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		//ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar4* d_inputImg = nullptr;
		float* d_matConv = nullptr;
		uchar4* d_output = nullptr;

		// Allocate arrays
		HANDLE_ERROR(cudaMalloc(&d_inputImg, sizeof(uchar4) * inputImg.size()));
		HANDLE_ERROR(cudaMalloc(&d_matConv, sizeof(float) * matConv.size()));
		HANDLE_ERROR(cudaMalloc(&d_output, sizeof(uchar4) * inputImg.size()));

		// Copy data from host to device
		HANDLE_ERROR(cudaMemcpy(d_inputImg, inputImg.data(), sizeof(uchar4) * inputImg.size(), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_matConv, matConv.data(), sizeof(float) * matConv.size(), cudaMemcpyHostToDevice));

		// Launch kernel
		const uint BLOCK_SIZE = 32;
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight/threadsPerBlock.y));
		convGPU<<<numBlocks, threadsPerBlock>>>(imgWidth, imgHeight, matSize, d_inputImg, d_matConv, d_output);
		HANDLE_ERROR(cudaDeviceSynchronize());

		// Copy data from device to host (output array)
		HANDLE_ERROR(cudaMemcpy(output.data(), d_output, sizeof(uchar4) * inputImg.size(), cudaMemcpyDeviceToHost));

		// Free arrays on device
		HANDLE_ERROR(cudaFree(d_inputImg));
		HANDLE_ERROR(cudaFree(d_matConv));
		HANDLE_ERROR(cudaFree(d_output));
	}
}
