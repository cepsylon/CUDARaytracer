#pragma once

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cuda_runtime.h>

class Camera
{
public:
	// Default constructor
	__device__ Camera() {}
	// Constructor
	__device__ Camera(const glm::vec3 & position, const glm::vec3 & right, const glm::vec3 & up, const glm::vec3 & center);

	// Gettors
	__device__ glm::vec3 position() const;
	__device__ glm::vec3 right() const;
	__device__ glm::vec3 up() const;
	__device__ glm::vec3 projection_center() const;

private:
	glm::vec3 mPosition;
	glm::vec3 mRight;
	glm::vec3 mUp;
	glm::vec3 mProjectionCenter;
};
