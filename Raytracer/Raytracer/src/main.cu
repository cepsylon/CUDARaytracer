#include "vec3.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb/stb_image_write.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void render_image(unsigned char * image_data, int width, int height)
{
	// Get coordinates from block and thread indices
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;
	int pixel_index = y * width * 3 + x * 3;

	// Cast ray
	vec3 color{ 1.0f };

	image_data[pixel_index] = static_cast<unsigned char>(color.r * 255.99f);
	image_data[pixel_index + 1] = static_cast<unsigned char>(color.g * 255.99f);
	image_data[pixel_index + 2] = static_cast<unsigned char>(color.b * 255.99f);
}

int main()
{
	int width = 500, height = 500;
	int buffer_size = width * height * 3;

	unsigned char * image_data = nullptr;
	cudaMallocManaged((void **)&image_data, buffer_size);

	// Compute needed blocks for the whole image
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);

	render_image <<<blocks,threads>>>(image_data, width, height);
	cudaDeviceSynchronize();

	stbi_write_png("MyOutput.png", width, height, 3, image_data, 0);
	return 0;
}
