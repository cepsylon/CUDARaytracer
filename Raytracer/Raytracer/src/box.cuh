#pragma once

#include "Surface.cuh"
#include "vector.cuh"

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


class Box : public Surface
{
public:
	__device__ Box(const Material & material, const glm::vec3 & position, const glm::vec3 & width, const glm::vec3 & height, const glm::vec3 & depth);

	__device__ bool collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const override;

	struct Plane
	{
		glm::vec3 mCenter;
		glm::vec3 mNormal;
	};

private:
	vector<Plane> mPlanes;
};
