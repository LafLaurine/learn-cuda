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
#include "convolution.hpp"

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

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		

		// Ex1
		/*
		{
			ChronoGPU chrGPU;
			std::cout << std::endl << "Start Ex1" << std::endl;
			chrGPU.start();
			ex1(inputImg, imgWidth, imgHeight, matConv, matSize, output);
			chrGPU.stop();
			std::cout 	<< "-> Ex1 Done : " << chrGPU.elapsedTime() << " ms" << std::endl;
			std::cout << "Compare with CPU" << std::endl;
			compareImages(output, resultCPU);
		}

		// Ex2
		{
			ChronoGPU chrGPU;
			std::cout << std::endl << "Start Ex2" << std::endl;
			chrGPU.start();
			ex2(inputImg, imgWidth, imgHeight, matConv, matSize, output);
			chrGPU.stop();
			std::cout 	<< "-> Ex2 Done : " << chrGPU.elapsedTime() << " ms" << std::endl;
			std::cout << "Compare with CPU" << std::endl;
			compareImages(output, resultCPU);
		}
		*/

		// Ex3
		{
			//ChronoGPU chrGPU;
			std::cout << std::endl << "Start Ex3" << std::endl;
			//chrGPU.start();
			ex3(inputImg, imgWidth, imgHeight, matConv, matSize, output);
			//chrGPU.stop();
			//std::cout 	<< "-> Ex3 Done : " << chrGPU.elapsedTime() << " ms" << std::endl;
			std::cout << "Compare with CPU" << std::endl;
			compareImages(output, resultCPU);
		}
	}
}
