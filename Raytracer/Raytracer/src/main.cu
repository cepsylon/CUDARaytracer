#include "debug.h"
#include "ray.cuh"
#include "sphere.cuh"
#include "scene.cuh"
#include "importer.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include <stb/stb_image_write.h>

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

const int width = 500;
const int height = 500;
const int buffer_size = width * height * 3;
const int half_width = width / 2;
const int half_height = height / 2;
const int shadow_sample_count = 100;
const int max_depth = 10;


__device__ float compute_refl_coefficient(const Ray & ray, const CollisionData & data, float ni)
{
	float cos_angle = glm::dot(-ray.direction(), data.mNormal);
	float ni_over_nt = ni / data.mMaterial.mRefractionIndex;
	float inside_sqrt = 1.0f - ni_over_nt * ni_over_nt * (1.0f - cos_angle * cos_angle);
	if (inside_sqrt < 0.0f)
		return 1.0f;
	float square_root = std::sqrt(inside_sqrt);
	float perpendicular = (ni_over_nt * cos_angle - square_root) / (ni_over_nt * cos_angle + square_root);
	float parallel = (cos_angle - ni_over_nt * square_root) / (cos_angle + ni_over_nt * square_root);
	return 0.5f * (perpendicular * perpendicular + parallel * parallel);
}

__device__ float compute_lit_percentage(const vector<Surface *> & surfaces, const PointLight & light, const glm::vec3 & intersection_point, curandState * random_state)
{
	CollisionData dummy;
	int shadow_count = 0;

	// Soft shadows
	for (unsigned current_sample = 0; current_sample < shadow_sample_count; ++current_sample)
	{
		Ray shadow_ray{ intersection_point, light.rand_pos(random_state) - intersection_point };

		// Check for collisions, if there's any, we are in shadow
		for (int j = 0; j < surfaces.size(); ++j)
		{
			if (surfaces[j]->collide(shadow_ray, 0.001f, 1.0f, dummy))
			{
				shadow_count++;
				break;
			}
		}
	}

	// Compute how much the surface is lit
	return 1.0f - static_cast<float>(shadow_count) / static_cast<float>(shadow_sample_count);
}

struct RayData
{
	Ray mRay;
	float mAttenuationInv;
	int mDepth;
};

__device__ glm::vec3 cast_ray(float x, float y, Scene * scene, curandState * random_state)
{
	float current_x = (static_cast<float>(x) + 0.5f - half_width) / half_width;
	float current_y = -(static_cast<float>(y) + 0.5f - half_height) / half_height;
	const Camera & camera = scene->camera();
	glm::vec3 pixel_position = camera.projection_center() + camera.right() * current_x + camera.up() * current_y;
	glm::vec3 final_color{ 0.0f };
	//float ni = 1.0f;

	// Vector to know how many rays we have left since recursion will lead to stackoverflow, we need an iterative mode
	// We start with the initial ray casted from the camera to the pixel coordinate
	vector<RayData> ray_stack;
	ray_stack.push_back(RayData{ Ray{ camera.position(), glm::normalize(pixel_position - camera.position()) }, 1.0f, 0 });

	while(ray_stack.empty() == false)
	{
		RayData ray_data = ray_stack.back();
		ray_stack.pop_back();

		// Check for collisions
		CollisionData collision_data;
		const vector<Surface *> & surfaces = scene->surfaces();
		for (int i = 0; i < surfaces.size(); ++i)
			surfaces[i]->collide(ray_data.mRay, 0.0f, collision_data.mT, collision_data);

		// Hit nothing, exit
		if (collision_data.mT == FLT_MAX)
			break;

		// Intersection point and reflected direction
		glm::vec3 intersection_point = ray_data.mRay.at(collision_data.mT);
		glm::vec3 reflected = glm::reflect(ray_data.mRay.direction(), collision_data.mNormal);

		// Ambient
		final_color += collision_data.mMaterial.mColor * scene->ambient() * ray_data.mAttenuationInv;

		// Local illumination
		const vector<PointLight> & lights = scene->lights();
		for (int i = 0; i < lights.size(); ++i)
		{
			// Check if surface is lit
			float lit = compute_lit_percentage(surfaces, lights[i], intersection_point, random_state);
			if (lit == 0.0f)
				continue;

			// Diffuse
			glm::vec3 to_light = glm::normalize(lights[i].position() - intersection_point);
			float cos_angle = glm::dot(to_light, collision_data.mNormal);
			if (cos_angle < 0.0f)
				continue;
			final_color += collision_data.mMaterial.mColor * lights[i].intensity() * cos_angle * ray_data.mAttenuationInv * lit;

			// Specular
			cos_angle = glm::dot(reflected, to_light);
			if (cos_angle > 0.0f)
			{
				float specular_value = powf(cos_angle, collision_data.mMaterial.mShininess) * collision_data.mMaterial.mSpecularCoefficient;
				final_color += lights[i].intensity() * specular_value * ray_data.mAttenuationInv * lit;
			}
		}

		// Reflection
		//float reflection_coefficient = compute_refl_coefficient(ray, collision_data, ni);
		int next_depth = ray_data.mDepth + 1;
		if (next_depth < max_depth && collision_data.mMaterial.mSpecularCoefficient)
		{
			Ray next_ray{ intersection_point + collision_data.mNormal * 0.001f, reflected };
			float next_attenuation = ray_data.mAttenuationInv * collision_data.mMaterial.mSpecularCoefficient;
			ray_stack.push_back(RayData{ next_ray, next_attenuation, next_depth });
		}
	}

	return glm::min(final_color, glm::vec3{ 1.0f });
}

__global__ void render_image(unsigned char * image_data, int width, int height, Scene * scene)
{
	// Get coordinates from block and thread indices
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;
	int pixel_index = y * width * 3 + x * 3;

	// Random state for pseudo random number generator
	curandState random_state;
	curand_init(1997, pixel_index, 0, &random_state);

	// Compute and store color
	glm::vec3 color = cast_ray(static_cast<float>(x), static_cast<float>(y), scene, &random_state);

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

	// Allocate memory in shared memory
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
