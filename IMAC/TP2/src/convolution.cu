#include "convolution.hpp"

#include <cassert>

namespace IMAC
{
    __device__ float cu_clampf(const float val, const float minVal, const float maxVal)
    {
        return min(maxVal, max(minVal, val));
    }

    __global__ void cu_ex1(const uint imgWidth, const uint imgHeight, const uint matSize, uchar4 *inputImg, float *matConv, uchar4 *output)
    {
        for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < imgHeight; h += blockDim.y * gridDim.y)
        {
            for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < imgWidth; w += blockDim.x * gridDim.x)
            {
                float3 sum = make_float3(0.f, 0.f, 0.f);
                // Apply convolution
                for (uint j = 0; j < matSize; ++j)
                {
                    for (uint i = 0; i < matSize; ++i)
                    {
                        int dX = w + i - matSize / 2;
                        int dY = h + j - matSize / 2;

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
                        const int idPixel = dY * imgWidth + dX;
                        sum.x += (float)inputImg[idPixel].x * matConv[idMat];
                        sum.y += (float)inputImg[idPixel].y * matConv[idMat];
                        sum.z += (float)inputImg[idPixel].z * matConv[idMat];
                    }
                }
                const int index = w + imgWidth * h;
                output[index].x = __float2uint_rd(cu_clampf(sum.x, 0.f, 255.f));
                output[index].y = __float2uint_rd(cu_clampf(sum.y, 0.f, 255.f));
                output[index].z = __float2uint_rd(cu_clampf(sum.z, 0.f, 255.f));
                output[index].w = 255;
            }
        }
    }

    void ex1(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, const std::vector<float> &matConv,
             const uint matSize, std::vector<uchar4> &output)
    {
        // 3 arrays for GPU
        uchar4 *d_inputImg = nullptr;
        float *d_matConv = nullptr;
        uchar4 *d_output = nullptr;

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
        dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight / threadsPerBlock.y));
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

    __global__ void cu_ex2(const uint imgWidth, const uint imgHeight, const uint matSize, uchar4 *inputImg, uchar4 *output)
    {
        for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < imgHeight; h += blockDim.y * gridDim.y)
        {
            for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < imgWidth; w += blockDim.x * gridDim.x)
            {
                float3 sum = make_float3(0.f, 0.f, 0.f);
                // Apply convolution
                for (uint j = 0; j < matSize; ++j)
                {
                    for (uint i = 0; i < matSize; ++i)
                    {
                        int dX = w + i - matSize / 2;
                        int dY = h + j - matSize / 2;

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
                        const int idPixel = dY * imgWidth + dX;
                        sum.x += (float)inputImg[idPixel].x * MAT_CONV[idMat];
                        sum.y += (float)inputImg[idPixel].y * MAT_CONV[idMat];
                        sum.z += (float)inputImg[idPixel].z * MAT_CONV[idMat];
                    }
                }
                const int index = w + imgWidth * h;
                output[index].x = __float2uint_rd(cu_clampf(sum.x, 0.f, 255.f));
                output[index].y = __float2uint_rd(cu_clampf(sum.y, 0.f, 255.f));
                output[index].z = __float2uint_rd(cu_clampf(sum.z, 0.f, 255.f));
                output[index].w = 255;
            }
        }
    }

    void ex2(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, const std::vector<float> &matConv,
             const uint matSize, std::vector<uchar4> &output)
    {
        assert(matConv.size() <= 225 && "matSize is too large for constant memory definition");

        // 2 arrays for GPU
        uchar4 *d_inputImg = nullptr;
        uchar4 *d_output = nullptr;

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
        dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight / threadsPerBlock.y));
        cu_ex2<<<numBlocks, threadsPerBlock>>>(imgWidth, imgHeight, matSize, d_inputImg, d_output);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Copy data from device to host (output array)
        HANDLE_ERROR(cudaMemcpy(output.data(), d_output, sizeof(uchar4) * inputImg.size(), cudaMemcpyDeviceToHost));

        // Free arrays on device
        HANDLE_ERROR(cudaFree(d_inputImg));
        HANDLE_ERROR(cudaFree(d_output));
    }

    texture<uchar4, cudaTextureType1D, cudaReadModeElementType> inputTex;

    __global__ void cu_ex3(const uint imgWidth, const uint imgHeight, const uint matSize, float *matConv, uchar4 *output)
    {
        for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < imgHeight; h += blockDim.y * gridDim.y)
        {
            for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < imgWidth; w += blockDim.x * gridDim.x)
            {
                float3 sum = make_float3(0.f, 0.f, 0.f);
                // Apply convolution
                for (uint j = 0; j < matSize; ++j)
                {
                    for (uint i = 0; i < matSize; ++i)
                    {
                        int dX = w + i - matSize / 2;
                        int dY = h + j - matSize / 2;

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
                        const int idPixel = dY * imgWidth + dX;
                        // tex1D is used for cudaArray. tex1Dfetch is for linear memory
                        uchar4 val = tex1Dfetch<uchar4>(inputTex, idPixel);
                        sum.x += (float)val.x * matConv[idMat];
                        sum.y += (float)val.y * matConv[idMat];
                        sum.z += (float)val.z * matConv[idMat];
                    }
                }
                const int index = w + imgWidth * h;
                output[index].x = __float2uint_rd(cu_clampf(sum.x, 0.f, 255.f));
                output[index].y = __float2uint_rd(cu_clampf(sum.y, 0.f, 255.f));
                output[index].z = __float2uint_rd(cu_clampf(sum.z, 0.f, 255.f));
                output[index].w = 255;
            }
        }
    }

    void ex3(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, const std::vector<float> &matConv,
             const uint matSize, std::vector<uchar4> &output)
    {
        // 3 arrays for GPU
        uchar4 *d_inputImg = nullptr;
        float *d_matConv = nullptr;
        uchar4 *d_output = nullptr;

        // Allocate arrays
        HANDLE_ERROR(cudaMalloc(&d_inputImg, sizeof(uchar4) * inputImg.size()));
        HANDLE_ERROR(cudaMalloc(&d_matConv, sizeof(float) * matConv.size()));
        HANDLE_ERROR(cudaMalloc(&d_output, sizeof(uchar4) * inputImg.size()));

        // Copy data from host to device
        HANDLE_ERROR(cudaMemcpy(d_inputImg, inputImg.data(), sizeof(uchar4) * inputImg.size(), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_matConv, matConv.data(), sizeof(float) * matConv.size(), cudaMemcpyHostToDevice));

        // Bind input image to texture memory
        HANDLE_ERROR(cudaBindTexture(0, inputTex, d_inputImg, sizeof(uchar4) * inputImg.size()));

        // Launch kernel
        const uint BLOCK_SIZE = 32;
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight / threadsPerBlock.y));
        cu_ex3<<<numBlocks, threadsPerBlock>>>(imgWidth, imgHeight, matSize, d_matConv, d_output);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Copy data from device to host (output array)
        HANDLE_ERROR(cudaMemcpy(output.data(), d_output, sizeof(uchar4) * inputImg.size(), cudaMemcpyDeviceToHost));

        // Free arrays on device
        HANDLE_ERROR(cudaFree(d_inputImg));
        HANDLE_ERROR(cudaFree(d_matConv));
        HANDLE_ERROR(cudaFree(d_output));
    }

    texture<uchar4, 2> inputTex2D;

    __global__ void cu_ex4(const uint imgWidth, const uint imgHeight, float *matConv, const uint matSize, uchar4 *output)
    {
        for (int h = threadIdx.y + blockDim.y * blockIdx.y; h < imgHeight; h += blockDim.y * gridDim.y)
        {
            for (int w = threadIdx.x + blockDim.x * blockIdx.x; w < imgWidth; w += blockDim.x * gridDim.x)
            {
                float3 sum = make_float3(0.f, 0.f, 0.f);
                // Apply convolution
                for (uint j = 0; j < matSize; ++j)
                {
                    for (uint i = 0; i < matSize; ++i)
                    {
                        int dX = w + i - matSize / 2;
                        int dY = h + j - matSize / 2;

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
                        uchar4 val = tex2D(inputTex2D, dX, dY);
                        sum.x += (float)val.x * matConv[idMat];
                        sum.y += (float)val.y * matConv[idMat];
                        sum.z += (float)val.z * matConv[idMat];
                    }
                }
                const int index = w + imgWidth * h;
                output[index].x = __float2uint_rd(cu_clampf(sum.x, 0.f, 255.f));
                output[index].y = __float2uint_rd(cu_clampf(sum.y, 0.f, 255.f));
                output[index].z = __float2uint_rd(cu_clampf(sum.z, 0.f, 255.f));
                output[index].w = 255;
            }
        }
    }

    void ex4(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, const std::vector<float> &matConv,
             const uint matSize, std::vector<uchar4> &output)
    {
        // 3 arrays for GPU
        uchar4 *d_inputImg = nullptr;
        float *d_matConv = nullptr;
        uchar4 *d_output = nullptr;
        size_t pitch;

        // Allocate arrays
        HANDLE_ERROR(cudaMallocPitch(&d_inputImg, &pitch, imgWidth * sizeof(uchar4), imgHeight));
        HANDLE_ERROR(cudaMalloc(&d_matConv, sizeof(float) * matConv.size()));
        HANDLE_ERROR(cudaMalloc(&d_output, sizeof(uchar4) * inputImg.size()));
        
        // Copy from host to device
        HANDLE_ERROR(cudaMemcpy2D(d_inputImg, pitch, inputImg.data(), sizeof(uchar4) * imgWidth, imgWidth * sizeof(uchar4), imgHeight, cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaBindTexture2D(NULL, inputTex2D, d_inputImg, inputTex2D.channelDesc, imgWidth, imgHeight, pitch));

        // launch kernel
        const uint BLOCK_SIZE = 32;
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks(ceil((float)imgWidth / threadsPerBlock.x), ceil((float)imgHeight / threadsPerBlock.y));
        cu_ex4<<<numBlocks, threadsPerBlock>>>(imgWidth, imgHeight, d_matConv, matSize, d_output);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Copy data from device to host (output array)
        HANDLE_ERROR(cudaMemcpy(output.data(), d_output, sizeof(uchar4) * inputImg.size(), cudaMemcpyDeviceToHost));

        // Free arrays on device
        HANDLE_ERROR(cudaUnbindTexture(inputTex2D));
        HANDLE_ERROR(cudaFree(d_output));
        HANDLE_ERROR(cudaFree(d_inputImg));
        HANDLE_ERROR(cudaFree(d_matConv));
    }
}