/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{

	__device__
    int cuda_getNumberToProcess(const uint arraySize)
    {
        return blockIdx.x == gridDim.x - 1 ? (arraySize - 1) % (2 * blockDim.x) + 1 : 2 * blockDim.x;
    }

	__device__
	void cuda_fillsharedArray(uint* sharedMemory, const uint* const dev_array, const uint size)
    {
        int localIdx = 2 * threadIdx.x;
        int globalIdx = localIdx + 2 + blockIdx.x * blockDim.x;
        if (globalIdx < size)
        {
            sharedMemory[localIdx] = dev_array[globalIdx];
            if (globalIdx + 1 < size)
            {
                sharedMemory[localIdx + 1] = dev_array[globalIdx + 1];
            }
        }
        __syncthreads();
    }

	__device__
	void cuda_fillVolatileSharedArray(volatile uint* sharedMemory, const uint* const dev_array, const uint size)
    {
        int localIdx = 2 * threadIdx.x;
        int globalIdx = localIdx + 2 + blockIdx.x * blockDim.x;
        if (globalIdx < size)
        {
            sharedMemory[localIdx] = dev_array[globalIdx];
            if (globalIdx + 1 < size)
            {
                sharedMemory[localIdx + 1] = dev_array[globalIdx + 1];
            }
        }
        __syncthreads();
    }
	
	// ==================================================== EX 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		extern __shared__ uint sharedMemory[];
		int localIdx = threadIdx.x;
		int globalIdx = localIdx + blockIdx.x * blockDim.x;

		if (globalIdx < size)
        {
            sharedMemory[localIdx] = dev_array[globalIdx];
        } else {
			sharedMemory[localIdx] = 0;
		}

		__syncthreads();

		for(unsigned int s = 1; s < blockDim.x; s *= 2)
		{
			const unsigned int sIndex = 2 * s * localIdx;
			int sNext = sIndex + s;
			if (sNext < blockDim.x)
			{
				sharedMemory[sIndex] = umax(sharedMemory[sIndex], sharedMemory[sNext]);
            }
			__syncthreads();
		}

		if(localIdx == 0) 
		{
			dev_partialMax[blockIdx.x] = sharedMemory[0];
		}
	}

	// ==================================================== EX 2, EX3
    __global__
    void maxReduce_ex23(const uint *const dev_array, const uint size, uint *const dev_partialMax)
    {
        extern __shared__ uint sharedMemory[];
		int localIdx = threadIdx.x;
		const int numberToProcess = cuda_getNumberToProcess(size);
		cuda_fillsharedArray(sharedMemory, dev_array, size);
		
		for(unsigned int s = 1; s < blockDim.x; s *= 2)
        {
			const unsigned int sIndex = localIdx;
            int numberToProcessStep = (blockDim.x - 1) / (2 * s) + 1;
            int sNext = sIndex + numberToProcessStep;
            if (2 * sIndex >= sNext || sNext >= blockDim.x )
            {
                break;
            }
            sharedMemory[sIndex] = umax(sharedMemory[sIndex], sharedMemory[sNext]);
            __syncthreads();
        }
        if(localIdx == 0) 
		{
			dev_partialMax[blockIdx.x] = sharedMemory[0];
		}
        __syncthreads();
    }

	__global__
    void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax, int warpSize)
    {
        extern __shared__ uint sharedMemory[];
        volatile uint* shared_array_volatile = sharedMemory;
        // in case size is not a power of two.
        const int numberToProcess = cuda_getNumberToProcess(size);

        cuda_fillVolatileSharedArray(shared_array_volatile, dev_array, size);

		for(unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            int numberToProcessStep = (numberToProcess - 1) / (2 * s) + 1;
            int sIdx = threadIdx.x;
            int sNext = sIdx + numberToProcessStep;
            if (2 * sIdx >= sNext || sNext >= numberToProcess)
            {
                break;
            }
            shared_array_volatile[sIdx] = umax(shared_array_volatile[sIdx], shared_array_volatile[sNext]);
            if (numberToProcessStep <= warpSize * 2)
            {
                __syncthreads();
            }
        }
        if (threadIdx.x == 0)
        {
            dev_partialMax[blockIdx.x] = shared_array_volatile[0];
        }
        __syncthreads();
    }

	// return a uint2 with x: dimBlock / y: dimGrid
	template<uint kernelType>
	uint2 configureKernel(const uint sizeArray)
	{
		uint2 dimBlockGrid; // x: dimBlock / y: dimGrid
		cudaDeviceProp prop;
        int device;
        HANDLE_ERROR(cudaGetDevice(&device));
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));

        unsigned long maxThreadsPerBlock = prop.maxThreadsPerBlock;
		const int totalNumberThreads = sizeArray / 2 + 1;

		// Configure number of threads/blocks
		switch(kernelType)
		{
			case KERNEL_EX1:
				dimBlockGrid.x = MAX_NB_THREADS; 
				dimBlockGrid.y = DEFAULT_NB_BLOCKS;
			break;
			case KERNEL_EX2:
				dimBlockGrid.y = (totalNumberThreads - 1) / maxThreadsPerBlock + 1;
				dimBlockGrid.x = (totalNumberThreads - 1) / dimBlockGrid.y + 1;
			break;
			case KERNEL_EX3:
				dimBlockGrid.y = (totalNumberThreads - 1) / maxThreadsPerBlock + 1;
				dimBlockGrid.x = (totalNumberThreads - 1) / dimBlockGrid.y + 1;
			break;
			case KERNEL_EX4:
				dimBlockGrid.y = (totalNumberThreads - 1) / maxThreadsPerBlock + 1;
				dimBlockGrid.x = (totalNumberThreads - 1) / dimBlockGrid.y + 1;
			break;
			case KERNEL_EX5:
				/// TODO EX 5
			break;
			default:
				throw std::runtime_error("Error configureKernel: unknown kernel type");
		}
		verifyDimGridBlock(dimBlockGrid.y, dimBlockGrid.x, sizeArray); // Are you reasonable ?
		
		return dimBlockGrid;
	}


    int getWarp()
    {
        cudaDeviceProp prop;
        int device;
        HANDLE_ERROR(cudaGetDevice(&device));
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
        return prop.warpSize;
    }

	// Launch kernel number 'kernelType' and return float2 for timing (x:device,y:host)    
	template<uint kernelType>
	float2 reduce(const uint nbIterations, const uint *const dev_array, const uint size, uint &result)
	{
		const uint2 dimBlockGrid = configureKernel<kernelType>(size); // x: dimBlock / y: dimGrid

		// Allocate arrays (host and device) for partial result
		std::vector<uint> host_partialMax(dimBlockGrid.y);
		const size_t bytesPartialMax = host_partialMax.size() * sizeof(uint);
		const size_t bytesSharedMem = dimBlockGrid.x * sizeof(uint);
		
		uint *dev_partialMax;
		HANDLE_ERROR(cudaMalloc((void**) &dev_partialMax, bytesPartialMax));

		std::cout 	<< "Computing on " << dimBlockGrid.y << " block(s) and " 
					<< dimBlockGrid.x << " thread(s) "
					<<"- shared mem size = " << bytesSharedMem << std::endl;
		
		ChronoGPU chrGPU;
		float2 timing = { 0.f, 0.f }; // x: timing GPU, y: timing CPU

		// Average timing on 'loop' iterations
		for (uint i = 0; i < nbIterations; ++i)
		{
			chrGPU.start();
			switch(kernelType) // Evaluated at compilation time
			{
				case KERNEL_EX1:
					std::cout << "Kernel 01 !" << std::endl;
					maxReduce_ex1<<<dimBlockGrid.y,dimBlockGrid.x,bytesSharedMem>>>(dev_array,size,dev_partialMax);
				break;
				case KERNEL_EX2:
					std::cout << "Kernel 02 !" << std::endl;
					maxReduce_ex23<<<dimBlockGrid.y, dimBlockGrid.x, 2*bytesSharedMem>>>(dev_array, size, dev_partialMax);
				break;
				case KERNEL_EX3:
					std::cout << "Kernel 03 !" << std::endl;
					maxReduce_ex23<<<dimBlockGrid.y, dimBlockGrid.x, 2*bytesSharedMem>>>(dev_array, size, dev_partialMax);
				break;
				case KERNEL_EX4:
					maxReduce_ex4<<<dimBlockGrid.y, dimBlockGrid.x, 2*bytesSharedMem>>>(dev_array,size,dev_partialMax,getWarp());
					std::cout << "Not implemented !" << std::endl;
				break;
				case KERNEL_EX5:
					/// TODO EX 5
					std::cout << "Not implemented !" << std::endl;
				break;
				default:
					cudaFree(dev_partialMax);
					throw("Error reduce: unknown kernel type.");
			}
			chrGPU.stop();
			timing.x += chrGPU.elapsedTime();
		}
		timing.x /= (float)nbIterations; // Stores time for device

		// Retrieve partial result from device to host
		HANDLE_ERROR(cudaMemcpy(host_partialMax.data(), dev_partialMax, bytesPartialMax, cudaMemcpyDeviceToHost));

		cudaFree(dev_partialMax);

		// Check for error
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			throw std::runtime_error(cudaGetErrorString(err));
		}
		
		ChronoCPU chrCPU;
		chrCPU.start();

		// Finish on host
		for (int i = 0; i < host_partialMax.size(); ++i)
		{
			result = std::max<uint>(result, host_partialMax[i]);
		}
		
		chrCPU.stop();

		timing.y = chrCPU.elapsedTime(); // Stores time for host
		
		return timing;
	}


	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR(cudaMalloc((void**)&dev_array, bytes));
		// Copy data from host to device
		HANDLE_ERROR(cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice));

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);
		
        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);
		
        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU);

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);
		
        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);
		
        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU);

		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);
		
        std::cout << " -> Done: ";
        printTiming(timing5);
		compare(res5, resCPU);

		// Free array on GPU
		cudaFree(dev_array);
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
