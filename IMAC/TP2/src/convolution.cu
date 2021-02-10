#include "convolution.hpp"

#include "common.hpp"

#include <cassert>

namespace IMAC
{
    __device__ float cu_clampf(const float val, const float minVal , const float maxVal)
    {
        return min(maxVal, max(minVal, val));
    }

    __global__ void cu_ex1(const uint imgWidth, const uint imgHeight, const uint matSize, uchar4* inputImg, float* matConv, uchar4* output)
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
            output[index].x = __float2uint_rd(cu_clampf( sum.x, 0.f, 255.f ));
            output[index].y = __float2uint_rd(cu_clampf( sum.y, 0.f, 255.f ));
            output[index].z = __float2uint_rd(cu_clampf( sum.z, 0.f, 255.f ));
            output[index].w = 255;
        }
    }

    void ex1(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, const std::vector<float> &matConv,
			const uint matSize, std::vector<uchar4> &output)
    {
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
		cu_ex1<<<numBlocks, threadsPerBlock>>>(imgWidth, imgHeight, matSize, d_inputImg, d_matConv, d_output);
		HANDLE_ERROR(cudaDeviceSynchronize());

		// Copy data from device to host (output array)
		HANDLE_ERROR(cudaMemcpy(output.data(), d_output, sizeof(uchar4) * inputImg.size(), cudaMemcpyDeviceToHost));

		// Free arrays on device
		HANDLE_ERROR(cudaFree(d_inputImg));
		HANDLE_ERROR(cudaFree(d_matConv));
		HANDLE_ERROR(cudaFree(d_output));
    }

    // Constant memory
    __constant__ float MAT_CONV[225]; // it is matConv.size()

    __global__ void cu_ex2(const uint imgWidth, const uint imgHeight, const uint matSize, uchar4* inputImg, uchar4* output)
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
                    sum.x += (float)inputImg[idPixel].x * MAT_CONV[idMat];
                    sum.y += (float)inputImg[idPixel].y * MAT_CONV[idMat];
                    sum.z += (float)inputImg[idPixel].z * MAT_CONV[idMat];
                }
            }
            output[index].x = __float2uint_rd(cu_clampf( sum.x, 0.f, 255.f ));
            output[index].y = __float2uint_rd(cu_clampf( sum.y, 0.f, 255.f ));
            output[index].z = __float2uint_rd(cu_clampf( sum.z, 0.f, 255.f ));
            output[index].w = 255;
        }
    }

    void ex2(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, const std::vector<float> &matConv,
			const uint matSize, std::vector<uchar4> &output)
    {
        assert(matConv.size() <= 225 && "matSize is too large for constant memory definition");

        // 2 arrays for GPU
		uchar4* d_inputImg = nullptr;
		uchar4* d_output = nullptr;

		// Allocate arrays
		HANDLE_ERROR(cudaMalloc(&d_inputImg, sizeof(uchar4) * inputImg.size()));
		HANDLE_ERROR(cudaMalloc(&d_output, sizeof(uchar4) * inputImg.size()));

		// Copy data from host to device
		HANDLE_ERROR(cudaMemcpy(d_inputImg, inputImg.data(), sizeof(uchar4) * inputImg.size(), cudaMemcpyHostToDevice));

        // Copy kernel to constant memory
        HANDLE_ERROR(cudaMemcpyToSymbol(MAT_CONV, matConv.data(), matConv.size() * sizeof(float)));

		// Launch kernel
		const uint BLOCK_SIZE = 32;
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight/threadsPerBlock.y));
		cu_ex2<<<numBlocks, threadsPerBlock>>>(imgWidth, imgHeight, matSize, d_inputImg, d_output);
		HANDLE_ERROR(cudaDeviceSynchronize());

		// Copy data from device to host (output array)
		HANDLE_ERROR(cudaMemcpy(output.data(), d_output, sizeof(uchar4) * inputImg.size(), cudaMemcpyDeviceToHost));

		// Free arrays on device
		HANDLE_ERROR(cudaFree(d_inputImg));
		HANDLE_ERROR(cudaFree(d_output));
    }

    __global__ void cu_ex3(cudaTextureObject_t inputTex)
    {
        uchar4 val = tex2D<uchar4>(inputTex, 0, 0);
        printf("Texture first pixel is %u %u %u %u \n", val.x, val.y, val.z, val.w);
    }

    void ex3(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, const std::vector<float> &matConv,
			const uint matSize, std::vector<uchar4> &output)
    {
        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(4, 4, 4, 4, cudaChannelFormatKindUnsigned); // uchar4
        cudaArray* d_inputImg;
        HANDLE_ERROR(cudaMallocArray(&d_inputImg, &channelDesc, imgWidth, imgHeight));

        // Copy to device memory
        HANDLE_ERROR(cudaMemcpyToArray(d_inputImg, 0, 0, inputImg.data(), inputImg.size() * sizeof(uchar4), cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = d_inputImg;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaTextureObject_t d_inputImgTex = 0;
        HANDLE_ERROR(cudaCreateTextureObject(&d_inputImgTex, &resDesc, &texDesc, NULL));

        // Launch kernel
		const uint BLOCK_SIZE = 32;
		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight/threadsPerBlock.y));
		cu_ex3<<<numBlocks, threadsPerBlock>>>(d_inputImgTex);
		HANDLE_ERROR(cudaDeviceSynchronize());

        // Destroy texture object
        HANDLE_ERROR(cudaDestroyTextureObject(d_inputImgTex));

        // Free device memory
        HANDLE_ERROR(cudaFreeArray(d_inputImg));
    }
}