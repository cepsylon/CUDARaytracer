#pragma once

#include "surface.cuh"

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

class Sphere : public Surface
{
public:
	__device__ Sphere(const Material & material, const glm::vec3 & position, float radius);

	__device__ bool collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const override;

	// Gettors
	__device__ glm::vec3 position() const;
	__device__ float radius() const;

private:
	glm::vec3 mPosition;
	float mRadius;
};
