#include "importer.cuh"

#include "debug.h"
#include "scene.cuh"
#include "sphere.cuh"
#include "box.cuh"
#include "ellipsoid.cuh"

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

__global__ void add_light_to_scene(Scene * scene, glm::vec3 position, glm::vec3 intensity)
{
	scene->add(PointLight{ position, intensity });
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
				glm::vec3 position, width, height, depth;

				// Position, width, height and depth
				sscanf_s(line.c_str(), "B (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f)",
					&position.x, &position.y, &position.z,
					&width.x, &width.y, &width.z,
					&height.x, &height.y, &height.z,
					&depth.x, &depth.y, &depth.z);

				// Material
				// Color, specular coefficient and shininess
				glm::vec3 color;
				float specular_coefficient, shininess;
				std::getline(file, line);
				sscanf_s(line.c_str(), "(%f,%f,%f) %f %f", &color.r, &color.g, &color.b, &specular_coefficient, &shininess);

				// Upload to GPU
				add_box_to_scene<<<1,1>>>(scene, Material{ color, specular_coefficient, shininess }, position, width, height, depth);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'C':
			{
				glm::vec3 position, right, up, center;
				
				// Projection center, right, up, distance to projection center
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
				glm::vec3 position, width, height, depth;

				// Position, width, height and depth
				sscanf_s(line.c_str(), "E (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f)",
					&position.x, &position.y, &position.z,
					&width.x, &width.y, &width.z,
					&height.x, &height.y, &height.z,
					&depth.x, &depth.y, &depth.z);

				// Material
				// Color, specular coefficient and shininess
				glm::vec3 color;
				float specular_coefficient, shininess;
				std::getline(file, line);
				sscanf_s(line.c_str(), "(%f,%f,%f) %f %f", &color.r, &color.g, &color.b, &specular_coefficient, &shininess);

				// Upload to GPU
				add_ellipsoid_to_scene<<<1,1>>>(scene, Material{ color, specular_coefficient, shininess }, position, width, height, depth);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'L':
			{
				glm::vec3 position, intensity;

				// Position and intensity
				sscanf_s(line.c_str(), "L (%f,%f,%f) (%f,%f,%f)",
					&position.x, &position.y, &position.z,
					&intensity.x, &intensity.y, &intensity.z);

				add_light_to_scene<<<1,1>>>(scene, position, intensity);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'S':
			{
				glm::vec3 position;
				float radius;

				// Position and radius
				sscanf_s(line.c_str(), "S (%f,%f,%f) %f",
					&position.x, &position.y, &position.z, &radius);

				// Material
				// Color, specular coefficient and shininess
				glm::vec3 color;
				float specular_coefficient, shininess;
				std::getline(file, line);
				sscanf_s(line.c_str(), "(%f,%f,%f) %f %f", &color.r, &color.g, &color.b, &specular_coefficient, &shininess);

				// Upload to GPU
				add_sphere_to_scene<<<1,1>>>(scene, Material{ color, specular_coefficient, shininess }, position, radius);
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

}
