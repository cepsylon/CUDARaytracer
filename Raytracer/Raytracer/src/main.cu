#include "ray.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb/stb_image_write.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

const int width = 500;
const int height = 500;
const int buffer_size = width * height * 3;
const int half_width = width / 2;
const int half_height = height / 2;

__device__ vec3 cast_ray(float x, float y)
{
	// Compute ray
	float current_x = (static_cast<float>(x) + 0.5f - half_width) / half_width;
	float current_y = -(static_cast<float>(y) + 0.5f - half_height) / half_height;
	vec3 direction = vec3{ 0.5f, 0.0f, 0.0f } * current_x + vec3{ 0.0f, 0.5f, 0.0f } * current_y;
	direction.z = -1.0f;
	Ray ray{ vec3{ 0.0f, 0.0f, 0.0f }, vec3::normalize(direction) };

	// Compute color
	vec3 final_color{ 0.0f };
	return final_color;
}

__global__ void render_image(unsigned char * image_data, int width, int height)
{
	// Get coordinates from block and thread indices
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;
	int pixel_index = y * width * 3 + x * 3;

	// Cast ray
	vec3 color = cast_ray(static_cast<float>(x), static_cast<float>(y));

	image_data[pixel_index] = static_cast<unsigned char>(color.r * 255.99f);
	image_data[pixel_index + 1] = static_cast<unsigned char>(color.g * 255.99f);
	image_data[pixel_index + 2] = static_cast<unsigned char>(color.b * 255.99f);
}

int main()
{
	// Allocate memory in GPU
	unsigned char * image_data = nullptr;
	cudaMallocManaged((void **)&image_data, buffer_size);

	// Compute needed blocks for the whole image
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);

	render_image <<<blocks,threads>>>(image_data, width, height);
	cudaDeviceSynchronize();

	stbi_write_png("MyOutput.png", width, height, 3, image_data, 0);

	// Free memory
	cudaFree(image_data);
	return 0;
}
