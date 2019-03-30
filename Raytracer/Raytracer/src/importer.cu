#include "importer.cuh"

#include "debug.h"
#include "vec3.cuh"
#include "scene.cuh"
#include "sphere.cuh"
#include "Box.cuh"

#include <fstream>
#include <string>

namespace importer
{

__global__ void add_sphere_to_scene(Scene * scene, Material material, vec3 position, float radius)
{
	scene->add(new Sphere{ material, position, radius });
}

__global__ void add_box_to_scene(Scene * scene, Material material, vec3 position, vec3 width, vec3 height, vec3 depth)
{
	scene->add(new Box{ material, position, width, height, depth });
}

__global__ void add_light_to_scene(Scene * scene, vec3 position, vec3 intensity)
{
	scene->add(PointLight{ position, intensity });
}

__global__ void set_scene_ambient(Scene * scene, vec3 ambient)
{
	scene->set_ambient(ambient);
}

__global__ void set_scene_camera(Scene * scene, vec3 position, vec3 right, vec3 up, vec3 center)
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
				vec3 ambient;
				sscanf_s(line.c_str(), "A (%f,%f,%f)", &ambient.x, &ambient.y, &ambient.z);

				// Upload to GPU
				set_scene_ambient<<<1,1>>>(scene, ambient);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'B':
			{
				vec3 position, width, height, depth;

				// Position, width, height and depth
				sscanf_s(line.c_str(), "B (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) (%f,%f,%f)",
					&position.x, &position.y, &position.z,
					&width.x, &width.y, &width.z,
					&height.x, &height.y, &height.z,
					&depth.x, &depth.y, &depth.z);

				// Material
				// Color, specular coefficient and shininess
				vec3 color;
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
				vec3 position, right, up, center;
				
				// Projection center, right, up, distance to projection center
				sscanf_s(line.c_str(), "C (%f,%f,%f) (%f,%f,%f) (%f,%f,%f) %f",
					&center.x, &center.y, &center.z,
					&right.x, &right.y, &right.z, 
					&up.x, &up.y, &up.z, 
					&position.x);

				// Compute position
				position = center + vec3::normalize(vec3::cross(right, up)) * position.x;

				// Upload to GPU
				set_scene_camera <<<1,1>>>(scene, position, right, up, center);
				CheckCUDAError(cudaGetLastError());
				CheckCUDAError(cudaDeviceSynchronize());
				break;
			}
			case 'L':
			{
				vec3 position, intensity;

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
				vec3 position;
				float radius;

				// Position and radius
				sscanf_s(line.c_str(), "S (%f,%f,%f) %f",
					&position.x, &position.y, &position.z, &radius);

				// Material
				// Color, specular coefficient and shininess
				vec3 color;
				float specular_coefficient, shininess;
				std::getline(file, line);
				sscanf_s(line.c_str(), "(%f,%f,%f) %f %f", &color.r, &color.g, &color.b, &specular_coefficient, &shininess);

				// Upload to GPU
				add_sphere_to_scene << <1, 1 >> > (scene, Material{ color, specular_coefficient, shininess }, position, radius);
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
