#pragma once

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class Ray
{
public:
	// Default constructor
	__device__ Ray() {}

	// Constructor
	__device__ Ray(const glm::vec3 & point, const glm::vec3 & direction);

	// Compute point at t
	__device__ glm::vec3 at(float t) const;

	// Gettors
	__device__ glm::vec3 point() const;
	__device__ glm::vec3 direction() const;
private:
	glm::vec3 mPoint;
	glm::vec3 mDirection;
};
