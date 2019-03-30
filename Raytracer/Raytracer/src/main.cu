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

	// Check for collisions
	CollisionData collision_data;
	const vector<Surface *> & surfaces = scene->surfaces();
	for (int i = 0; i < surfaces.size(); ++i)
		surfaces[i]->collide(ray, 0.0f, collision_data.mT, collision_data);

	// No need to color pixel
	if (collision_data.mT == FLT_MAX)
		return vec3{ 0.0f };


	// Shadow check
	vec3 final_color{ collision_data.mMaterial.mColor * scene->ambient() };
	vec3 collision_point = ray.at(collision_data.mT);
	vec3 reflected = vec3::reflect(ray.direction(), collision_data.mNormal);
	const vector<PointLight> & lights = scene->lights();
	for (int i = 0; i < lights.size(); ++i)
	{
		CollisionData dummy;
		ray = Ray{ collision_point, lights[i].position() - collision_point };
		int shadow_count = 0;
		
		// Check for collisions, if there's any, we are in shadow
		for(int j = 0; j < surfaces.size(); ++j)
		{
			if (surfaces[j]->collide(ray, 0.001f, 1.0f, dummy))
				shadow_count++;
		}
		
		// In shadow
		if (shadow_count)
			continue;
		
		// Diffuse
		vec3 to_light = vec3::normalize(ray.direction());
		float cos_angle = vec3::dot(to_light, collision_data.mNormal);
		if (cos_angle < 0.0f)
			continue;
		final_color += collision_data.mMaterial.mColor * lights[i].intensity() * cos_angle;

		// Specular
		cos_angle = vec3::dot(reflected, to_light);
		if (cos_angle > 0.0f)
			final_color += lights[i].intensity() * powf(cos_angle, collision_data.mMaterial.mShininess) * collision_data.mMaterial.mSpecularCoefficient;
	}

	return vec3::min(final_color, vec3{ 1.0f });
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
