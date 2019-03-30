#include "importer.cuh"

#include "debug.h"
#include "scene.cuh"
#include "sphere.cuh"
#include "box.cuh"
#include "ellipsoid.cuh"
#include "polygon.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <fstream>
#include <string>

namespace importer
{

__global__ void add_sphere_to_scene(Scene * scene, Material material, glm::vec3 position, float radius)
{
	scene->add(new Sphere{ material, position, radius });
}

__global__ void add_box_to_scene(Scene * scene, Material material, glm::vec3 position, glm::vec3 width, glm::vec3 height, glm::vec3 depth)
{
	scene->add(new Box{ material, position, width, height, depth });
}

__global__ void add_ellipsoid_to_scene(Scene * scene, Material material, glm::vec3 position, glm::vec3 width, glm::vec3 height, glm::vec3 depth)
{
	scene->add(new Ellipsoid{ material, position, width, height, depth });
}

__global__ void add_polygon_to_scene(Scene * scene, Material material, glm::vec3 * data, int count)
{
	scene->add(new Polygon{ material, data, count });
}

__global__ void add_light_to_scene(Scene * scene, glm::vec3 position, glm::vec3 intensity, float radius)
{
	scene->add(PointLight{ position, intensity, radius });
}

__global__ void set_scene_ambient(Scene * scene, glm::vec3 ambient)
{
	scene->set_ambient(ambient);
}

__global__ void set_scene_camera(Scene * scene, glm::vec3 position, glm::vec3 right, glm::vec3 up, glm::vec3 center)
{
	scene->set_camera(Camera{ position, right, up, center });
}

void import_scene(const char * path, Scene * scene)
{
	std::ifstream file{ path };
	if (file.is_open())
	{
		// Parse
		std::string line;
		while (std::getline(file, line))
		{
			switch (line[0])
			{
			case 'A':
			{
				// Parse
				glm::vec3 ambient;
				sscanf_s(line.c_str(), "A (%f,%f,%f)", &ambient.x, &ambient.y, &ambient.z);

				// Upload to GPU
				set_scene_ambient<<<1,1>>>(scene, ambient);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'B':
			{
				// Parse position, width, height and depth
				glm::vec3 position, width, height, depth;
				sscanf_s(line.c_str(), "B (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f)",
					&position.x, &position.y, &position.z,
					&width.x, &width.y, &width.z,
					&height.x, &height.y, &height.z,
					&depth.x, &depth.y, &depth.z);

				// Parse material
				std::getline(file, line);
				Material material = import_material(line);

				// Upload to GPU
				add_box_to_scene<<<1,1>>>(scene, material, position, width, height, depth);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'C':
			{
				// Parse projection center, right, up, distance to projection center
				glm::vec3 position, right, up, center;
				sscanf_s(line.c_str(), "C (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) %f",
					&center.x, &center.y, &center.z,
					&right.x, &right.y, &right.z, 
					&up.x, &up.y, &up.z, 
					&position.x);

				// Compute position
				position = center + glm::normalize(glm::cross(right, up)) * position.x;

				// Upload to GPU
				set_scene_camera<<<1,1>>>(scene, position, right, up, center);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'E':
			{
				// Parse position, width, height and depth
				glm::vec3 position, width, height, depth;
				sscanf_s(line.c_str(), "E (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f)",
					&position.x, &position.y, &position.z,
					&width.x, &width.y, &width.z,
					&height.x, &height.y, &height.z,
					&depth.x, &depth.y, &depth.z);

				// Parse material
				std::getline(file, line);
				Material material = import_material(line);

				// Upload to GPU
				add_ellipsoid_to_scene<<<1,1>>>(scene, material, position, width, height, depth);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'L':
			{
				// Parse position and intensity
				glm::vec3 position, intensity;
				float radius;
				sscanf_s(line.c_str(), "L (%f,%f,%f) (%f,%f,%f) %f",
					&position.x, &position.y, &position.z,
					&intensity.x, &intensity.y, &intensity.z,
					&radius);

				// Upload to GPU
				add_light_to_scene<<<1,1>>>(scene, position, intensity, radius);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'P':
			{
				// Parse vertex count
				int count = 0;
				sscanf_s(line.c_str(), "P %d", &count);

				// Shared memory to pass vertices to gpu
				glm::vec3 * vertices = nullptr;
				CheckCUDAError(cudaMallocManaged((void **)&vertices, sizeof(glm::vec2) * count));

				// Parse vertices
				size_t start = line.find_first_of("(");
				size_t end = line.find_first_of(")", start) + 1;
				for (int i = 0; i < count; ++i)
				{
					std::string vertex = line.substr(start, end - start);
					sscanf_s(vertex.c_str(), "(%f,%f,%f)", &vertices[i].x, &vertices[i].y, &vertices[i].z);

					start = line.find_first_of("(", end);
					end = line.find_first_of(")", start) + 1;
				}

				// Parse material
				std::getline(file, line);
				Material material = import_material(line);

				// Upload to GPU
				add_polygon_to_scene<<<1,1>>>(scene, material, vertices, count);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());

				// Free memory
				CheckCUDAError(cudaFree(vertices));
				break;
			}
			case 'S':
			{
				glm::vec3 position;
				float radius;

				// Parse position and radius
				sscanf_s(line.c_str(), "S (%f,%f,%f) %f",
					&position.x, &position.y, &position.z, &radius);

				// Parse material
				std::getline(file, line);
				Material material = import_material(line);

				// Upload to GPU
				add_sphere_to_scene<<<1,1>>>(scene, material, position, radius);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			default:
				break;
			}
		}
	}
}

Material import_material(const std::string & line)
{
	glm::vec3 color, attenuation;
	float specular_coefficient, shininess, permittivity, permeability;
	sscanf_s(line.c_str(), "(%f,%f,%f) %f %f (%f,%f,%f) %f %f",
		&color.r, &color.g, &color.b,
		&specular_coefficient, &shininess,
		&attenuation.x, &attenuation.y, &attenuation.z,
		&permittivity, &permeability);

	float refraction_index = std::sqrt(permittivity * permeability);
	return Material{ color, attenuation, specular_coefficient, shininess, permeability, refraction_index };
}

}
