#include "importer.cuh"

#include "debug.h"
#include "vec3.cuh"
#include "sphere.cuh"
#include "scene.cuh"

#include <fstream>
#include <string>

namespace importer
{

__global__ void add_sphere_to_scene(Scene * scene, vec3 color, vec3 position, float radius)
{
	scene->add(new Sphere{ color, position, radius });
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
			case 'S':
			{
				vec3 position, color;
				float radius;

				// Position and radius
				sscanf_s(line.c_str(), "S (%f,%f,%f) %f",
					&position.x, &position.y, &position.z, &radius);

				// Material
				std::getline(file, line);
				sscanf_s(line.c_str(), "(%f,%f,%f)", &color.r, &color.g, &color.b);

				add_sphere_to_scene<<<1,1>>>(scene, color, position, radius);
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
