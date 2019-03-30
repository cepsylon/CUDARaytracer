#pragma once

#include "surface.cuh"
#include "vector.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

class Polygon : public Surface
{
public:
	// Constructor
	__device__ Polygon(const Material & material, glm::vec3 * vertices, int count);

	__device__ bool collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const override;

	// Gettors
	__device__ const vector<glm::vec3> & vertices() const;

private:
	vector<glm::vec3> mVertices;
	glm::vec3 mNormal;
};
