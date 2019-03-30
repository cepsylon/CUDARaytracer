#pragma once

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class PointLight
{
public:
	// Default constructor
	__device__ PointLight() {}
	// Constructor
	__device__ PointLight(const glm::vec3 & position, const glm::vec3 & intensity);

	// Gettors
	__device__ glm::vec3 position() const;
	__device__ glm::vec3 intensity() const;

private:
	glm::vec3 mPosition;
	glm::vec3 mIntensity;
};
