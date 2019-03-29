#include "debug.h"
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "scene.cuh"
#include "importer.cuh"

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

__device__ vec3 cast_ray(float x, float y, Scene * scene)
{
	// Compute ray
	float current_x = (static_cast<float>(x) + 0.5f - half_width) / half_width;
	float current_y = -(static_cast<float>(y) + 0.5f - half_height) / half_height;
	const Camera & camera = scene->camera();
	vec3 pixel_position = camera.projection_center() + camera.right() * current_x + camera.up() * current_y;
	Ray ray{ camera.position(), vec3::normalize(pixel_position - camera.position()) };

	// Compute color
	vec3 final_color{ 0.0f };
	CollisionData collision_data;
	const vector<Surface *> & surfaces = scene->surfaces();
	for (int i = 0; i < surfaces.size(); ++i)
	{
		if (surfaces[i]->collide(ray, 0.0f, collision_data.mT, collision_data))
			final_color += collision_data.mColor;
	}
	return final_color;
}

__global__ void render_image(unsigned char * image_data, int width, int height, Scene * scene)
{
	// Get coordinates from block and thread indices
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;
	int pixel_index = y * width * 3 + x * 3;

	// Compute and store color
	vec3 color = cast_ray(static_cast<float>(x), static_cast<float>(y), scene);

	image_data[pixel_index] = static_cast<unsigned char>(color.r * 255.99f);
	image_data[pixel_index + 1] = static_cast<unsigned char>(color.g * 255.99f);
	image_data[pixel_index + 2] = static_cast<unsigned char>(color.b * 255.99f);
}

__global__ void initialize_scene(Scene * scene)
{
	new (scene) Scene{};
}

__global__ void populate_scene(Scene * scene)
{
	vec3 color{ 1.0f, 0.0f, 0.0f };
	vec3 center{ 0.0f, 0.0f, -5.0f };
	float radius = 0.5f;
	scene->add(new Sphere{ color, center, radius });
	color = vec3{ 0.0f, 1.0f, 0.0f };
	center.x = 1.0f;
	scene->add(new Sphere{ color, center, radius });
	color = vec3{ 0.0f, 0.0f, 1.0f };
	center.y = 1.0f;
	scene->add(new Sphere{ color, center, radius });
	color = vec3{ 1.0f, 1.0f, 0.0f };
	center.x = 0.0f;
	scene->add(new Sphere{ color, center, radius });
}

__global__ void destroy_scene(Scene * scene)
{
	scene->~Scene();
}

int main()
{
	// Scene creation
	Scene * scene = nullptr;
	CheckCUDAError(cudaMalloc((void **)&scene, sizeof(Scene)));
	initialize_scene<<<1,1>>>(scene);
	CheckCUDAError(cudaGetLastError());
	CheckCUDAError(cudaDeviceSynchronize());
	importer::import_scene("scene.txt", scene);

	// Allocate memory in GPU
	unsigned char * image_data = nullptr;
	CheckCUDAError(cudaMallocManaged((void **)&image_data, buffer_size));

	// Compute needed blocks for the whole image
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);

	// Render image
	render_image<<<blocks,threads>>>(image_data, width, height, scene);
	CheckCUDAError(cudaGetLastError());
	CheckCUDAError(cudaDeviceSynchronize());

	// Store color
	stbi_write_png("MyOutput.png", width, height, 3, image_data, 0);

	// Free memory
	cudaFree(image_data);
	destroy_scene<<<1,1>>>(scene);
	CheckCUDAError(cudaGetLastError());
	CheckCUDAError(cudaDeviceSynchronize());
	cudaFree(scene);
	return 0;
}
