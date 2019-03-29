#include "debug.h"

#include <cstdlib>	// std::exit
#include <iostream>	// std::cout, std::endl

void check_cuda_error(cudaError_t error, const char * func_name, const char * file, const int line)
{
	if (error)
	{
		// Print error
		std::cout << "CUDA error: " << error << " at file " << file << " line " << line << " function " << func_name << std::endl;

		// Reset device
		cudaDeviceReset();
		std::exit(1);
	}
}
