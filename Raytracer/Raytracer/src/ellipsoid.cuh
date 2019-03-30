#pragma once

#include "surface.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

class Ellipsoid : public Surface
{
public:
	__device__ Ellipsoid(const Material & material, const glm::vec3 & position, const glm::vec3 & width, const glm::vec3 & height, const glm::vec3 & depth);

	__device__ bool collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const override;

private:
	glm::vec3 mPosition;
	glm::vec3 mWidth;
	glm::vec3 mHeight;
	glm::vec3 mDepth;
};
