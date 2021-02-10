/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"
#include <algorithm>

#define BLOCK_SIZE 32
#define KERNEL_SIZE 16

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
				if (std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
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
	
	__global__ void applyConvolution(const unsigned char* dev_input, const uint imgWidth, const uint imgHeight, const uint matSize, float* dev_matConv, unsigned char* dev_output)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int index = (idy * imgWidth + idx) * 4;
		if (idx < imgWidth && idy < imgHeight){
			float3 sum = make_float3(0.f,0.f,0.f);
			
			// Apply convolution
			for (uint j = 0; j < matSize; j++) 
			{
				for (uint i = 0; i < matSize; i++) 
				{
					int dX = idx + i - matSize / 2;
					int dY = idy + j - matSize / 2;

					// Handle borders
					if (dX < 0) 
						dX = 0;

					if (dX >= imgWidth) 
						dX = imgWidth - 1;

					if (dY < 0) 
						dY = 0;

					if (dY >= imgHeight) 
						dY = imgHeight - 1;
					
					const int idMat = j * matSize + i;
					const int idPixel = (dY * imgWidth + dX) * 4;
					sum.x += (float)dev_input[idPixel] * dev_matConv[idMat];
					sum.y += (float)dev_input[idPixel+1] * dev_matConv[idMat];
					sum.z += (float)dev_input[idPixel+2] * dev_matConv[idMat];
				}
			}
			dev_output[index] = (uchar)max(0.f,min(255.f,sum.x));
			dev_output[index+1] = (uchar)max(0.f,min(255.f,sum.y));
			dev_output[index+2] = (uchar)max(0.f,min(255.f,sum.z));
			dev_output[index+3] = 255;
		}
	}

	__device__ __constant__ float dev_matConv[KERNEL_SIZE * KERNEL_SIZE];

	__global__ void applyConvolutionv2(const unsigned char* dev_input, const uint imgWidth, const uint imgHeight,  const uint matSize, unsigned char* dev_output)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int index = (idy * imgWidth + idx) * 4;
		if (idx < imgWidth && idy < imgHeight){
			float3 sum = make_float3(0.f,0.f,0.f);
			for (uint j = 0; j < matSize; j++) 
			{
				for (uint i = 0; i < matSize; i++) 
				{
					int dX = idx + i - matSize / 2;
					int dY = idy + j - matSize / 2;

					// Handle borders
					if (dX < 0) 
						dX = 0;

					if (dX >= imgWidth) 
						dX = imgWidth - 1;

					if (dY < 0) 
						dY = 0;

					if (dY >= imgHeight) 
						dY = imgHeight - 1;
					
					const int idMat = j * matSize + i;
					const int idPixel = (dY * imgWidth + dX) * 4;
					sum.x += (float)dev_input[idPixel] * dev_matConv[idMat];
					sum.y += (float)dev_input[idPixel+1] * dev_matConv[idMat];
					sum.z += (float)dev_input[idPixel+2] * dev_matConv[idMat];
				}
			}
			dev_output[index] = (uchar)max(0.f,min(255.f,sum.x));
			dev_output[index+1] = (uchar)max(0.f,min(255.f,sum.y));
			dev_output[index+2] = (uchar)max(0.f,min(255.f,sum.z));
			dev_output[index+3] = 255;
		}
	}

	texture<uchar> texRef;
	texture<uchar4,2> tex2DRef;

	__global__ void applyConvolutionv3(const uint imgWidth, const uint imgHeight,  const uint matSize, unsigned char* dev_output)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int index = (idy * imgWidth + idx) * 4;
		if (idx < imgWidth && idy < imgHeight){
			float3 sum = make_float3(0.f,0.f,0.f);
			for (uint j = 0; j < matSize; j++) 
			{
				for (uint i = 0; i < matSize; i++) 
				{
					int dX = idx + i - matSize / 2;
					int dY = idy + j - matSize / 2;

					// Handle borders
					if (dX < 0) 
						dX = 0;

					if (dX >= imgWidth) 
						dX = imgWidth - 1;

					if (dY < 0) 
						dY = 0;

					if (dY >= imgHeight) 
						dY = imgHeight - 1;
					
					const int idMat = j * matSize + i;
					const int idPixel = (dY * imgWidth + dX) * 4;
					sum.x += (float)tex1Dfetch(texRef,idPixel) * dev_matConv[idMat];
					sum.y += (float)tex1Dfetch(texRef,idPixel+1) * dev_matConv[idMat];
					sum.z += (float)tex1Dfetch(texRef,idPixel+2) * dev_matConv[idMat];
				}
			}
			dev_output[index] = (uchar)max(0.f,min(255.f,sum.x));
			dev_output[index+1] = (uchar)max(0.f,min(255.f,sum.y));
			dev_output[index+2] = (uchar)max(0.f,min(255.f,sum.z));
			dev_output[index+3] = 255.f;
		}
	}

	__global__ void applyConvolutionv4(const uint imgWidth, const uint imgHeight,  const uint matSize, unsigned char* dev_output)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int index = (idy * imgWidth + idx) * 4;
		if (idx < imgWidth && idy < imgHeight){
			float4 sum = make_float4(0.f,0.f,0.f,0.f);
			for (uint j = 0; j < matSize; j++) 
			{
				for (uint i = 0; i < matSize; i++) 
				{
					int dX = idx + i - matSize / 2;
					int dY = idy + j - matSize / 2;

					// Handle borders
					if (dX < 0) 
						dX = 0;

					if (dX >= imgWidth) 
						dX = imgWidth - 1;

					if (dY < 0) 
						dY = 0;

					if (dY >= imgHeight) 
						dY = imgHeight - 1;
					
					const int idMat = j * matSize + i;
					const uchar4 c = tex2D(tex2DRef,dX,dY);
					sum.x += (float)(c.x) * dev_matConv[idMat];
					sum.y += (float)(c.y) * dev_matConv[idMat];
					sum.z += (float)(c.z) * dev_matConv[idMat];
				}
			}
			dev_output[index] = (uchar)max(0.f,min(255.f,sum.x));
			dev_output[index+1] = (uchar)max(0.f,min(255.f,sum.y));
			dev_output[index+2] = (uchar)max(0.f,min(255.f,sum.z));
			dev_output[index+3] = 255.f;
		}
	}

    void studentJob1(const std::vector<uchar4> &inputImg, // Input image
		const uint imgWidth, const uint imgHeight, // Image size
		const std::vector<float> &matConv, // Convolution matrix (square)
		const uint matSize, // Matrix size (width or height)
		std::vector<uchar4> &output // Output image
		)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar* dev_input = NULL;
		uchar* dev_output = NULL;
		float* dev_matConv = NULL;
		chrGPU.start();
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight/threadsPerBlock.y));

		// allocate GPU buffers
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, imgWidth * imgHeight * 4 * sizeof(uchar)));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, imgWidth * imgHeight * 4 * sizeof(uchar)));
		HANDLE_ERROR(cudaMalloc((void**)&dev_matConv, matSize * matSize * sizeof(float)));

		// Copy data from host to device (input arrays) 
		HANDLE_ERROR(cudaMemcpy(dev_input, inputImg.data(), (imgWidth * imgHeight * 4) * sizeof(uchar), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(dev_matConv, matConv.data(), matSize * matSize * sizeof(float), cudaMemcpyHostToDevice));

		//launch kernel
		applyConvolution<<<numBlocks,threadsPerBlock>>>(dev_input,imgWidth,imgHeight,matSize,dev_matConv,dev_output);

		chrGPU.stop();
		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		HANDLE_ERROR(cudaDeviceSynchronize());
		
		// Copy data from device to host (output array)
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, (imgWidth * imgHeight * 4) * sizeof(uchar), cudaMemcpyDeviceToHost));

		// Free arrays on device
		HANDLE_ERROR(cudaFree(dev_output));
		HANDLE_ERROR(cudaFree(dev_matConv));
		HANDLE_ERROR(cudaFree(dev_input));
	}

	void studentJob2(const std::vector<uchar4> &inputImg, // Input image
		const uint imgWidth, const uint imgHeight, // Image size
		const std::vector<float> &matConv, // Convolution matrix (square)
		const uint matSize, // Matrix size (width or height)
		std::vector<uchar4> &output // Output image
		)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar* dev_input = NULL;
		uchar* dev_output = NULL;

		chrGPU.start();
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight/threadsPerBlock.y));

		// allocate GPU buffers
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, imgWidth * imgHeight * 4 * sizeof(uchar)));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, imgWidth * imgHeight * 4 * sizeof(uchar)));

		// Copy data from host to device (input arrays) 
		HANDLE_ERROR(cudaMemcpyToSymbol(dev_matConv, matConv.data(), matSize*matSize*sizeof(float)));
		HANDLE_ERROR(cudaMemcpy(dev_input, inputImg.data(), (imgWidth * imgHeight * 4) * sizeof(uchar), cudaMemcpyHostToDevice));

		//launch kernel
		applyConvolutionv2<<<numBlocks,threadsPerBlock>>>(dev_input,imgWidth,imgHeight,matSize,dev_output);

		chrGPU.stop();
		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		HANDLE_ERROR(cudaDeviceSynchronize());
		
		// Copy data from device to host (output array)
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, (imgWidth * imgHeight * 4) * sizeof(uchar), cudaMemcpyDeviceToHost));

		// Free arrays on device
		HANDLE_ERROR(cudaFree(dev_output));
		HANDLE_ERROR(cudaFree(dev_input));
	}

	void studentJob3(const std::vector<uchar4> &inputImg, // Input image
		const uint imgWidth, const uint imgHeight, // Image size
		const std::vector<float> &matConv, // Convolution matrix (square)
		const uint matSize, // Matrix size (width or height)
		std::vector<uchar4> &output // Output image
		)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		uchar* dev_input = NULL;
		uchar* dev_output = NULL;

		chrGPU.start();
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight/threadsPerBlock.y));

		// allocate GPU buffers
		HANDLE_ERROR(cudaMalloc((void**)&dev_input, imgWidth * imgHeight * 4 * sizeof(uchar)));
		HANDLE_ERROR(cudaMalloc((void**)&dev_output, imgWidth * imgHeight * 4 * sizeof(uchar)));

		// Copy data from host to device (input arrays) 
		HANDLE_ERROR(cudaMemcpyToSymbol(dev_matConv, matConv.data(), matSize*matSize*sizeof(float)));
		HANDLE_ERROR(cudaMemcpy(dev_input, inputImg.data(), (imgWidth * imgHeight * 4) * sizeof(uchar), cudaMemcpyHostToDevice));
		// bind texture
		HANDLE_ERROR(cudaBindTexture(NULL, texRef, dev_input,  imgWidth * imgHeight * 4 * sizeof(uchar)));

		//launch kernel
		applyConvolutionv3<<<numBlocks,threadsPerBlock>>>(imgWidth,imgHeight,matSize,dev_output);

		chrGPU.stop();
		std::cout << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		HANDLE_ERROR(cudaDeviceSynchronize());

		// Copy data from device to host (output array)
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, (imgWidth * imgHeight * 4) * sizeof(uchar), cudaMemcpyDeviceToHost));

		// Free arrays on device
		HANDLE_ERROR(cudaUnbindTexture(texRef));
		HANDLE_ERROR(cudaFree(dev_output));
		HANDLE_ERROR(cudaFree(dev_input));
	}

    void studentJob4(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
                    std::vector<uchar4> &output // Output image
					)
	{
		// 3 arrays for GPU
		uchar* dev_input = NULL;
		uchar* dev_output = NULL;
		size_t pitch;

		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight/threadsPerBlock.y));

		// allocate GPU buffers
		HANDLE_ERROR(cudaMallocPitch((void**)&dev_input, &pitch, (imgWidth) * sizeof(uchar) * 4, imgHeight));

		HANDLE_ERROR(cudaMemcpy2D(dev_input,pitch,inputImg.data(),  sizeof(uchar) * 4 *imgWidth,imgWidth * sizeof(uchar) * 4, imgHeight, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpyToSymbol(dev_matConv, matConv.data(), matSize*matSize*sizeof(float)));
		
		HANDLE_ERROR(cudaBindTexture2D(NULL, tex2DRef, dev_input, tex2DRef.channelDesc, imgWidth, imgHeight, pitch));

		//tex2DRef.normalized = false;  // don't use normalized values                   

		HANDLE_ERROR(cudaMalloc(&dev_output, 4*imgWidth*imgHeight*sizeof(uchar)));

		//launch kernel
		applyConvolutionv4<<<numBlocks,threadsPerBlock>>>(imgWidth,imgHeight,matSize,dev_output);

		HANDLE_ERROR(cudaDeviceSynchronize());
 		// Copy data from device to host (output array)
		HANDLE_ERROR(cudaMemcpy(output.data(), dev_output, 4 * imgHeight * imgWidth * sizeof(uchar), cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaUnbindTexture(tex2DRef));
		// Free arrays on device
		HANDLE_ERROR(cudaFree(dev_output));
		HANDLE_ERROR(cudaFree(dev_input));
	}

	void studentJob(const std::vector<uchar4> &inputImg, // Input image
		const uint imgWidth, const uint imgHeight, // Image size
		const std::vector<float> &matConv, // Convolution matrix (square)
		const uint matSize, // Matrix size (width or height)
		const std::vector<uchar4> &resultCPU, // Just for comparison
		std::vector<uchar4> &output // Output image
		)
	{
		std::cout << "------------------" << std::endl;
		std::cout << "GPU " << std::endl;
		std::cout << "Student Job 1 : " << std::endl;
		studentJob1(inputImg,imgWidth,imgHeight,matConv,matSize,output);
		compareImages(resultCPU,output);
		std::cout << "Student Job 2 : " << std::endl;
		studentJob2(inputImg,imgWidth,imgHeight,matConv,matSize,output);
		compareImages(resultCPU,output);

		std::cout << "Student Job 3 : " << std::endl;
		studentJob3(inputImg,imgWidth,imgHeight,matConv,matSize,output);
		compareImages(resultCPU,output);
		std::cout << "Student Job 4 : " << std::endl;
		studentJob4(inputImg,imgWidth,imgHeight,matConv,matSize,output);
		compareImages(resultCPU,output);
	}
}
