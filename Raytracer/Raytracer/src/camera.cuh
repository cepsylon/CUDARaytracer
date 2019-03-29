#pragma once

#include "vec3.cuh"

#include <cuda_runtime.h>

class Camera
{
public:
	// Default constructor
	__device__ Camera() {}
	// Constructor
	__device__ Camera(const vec3 & position, const vec3 & right, const vec3 & up, const vec3 & center);

	// Gettors
	__device__ vec3 position() const;
	__device__ vec3 right() const;
	__device__ vec3 up() const;
	__device__ vec3 projection_center() const;

private:
	vec3 mPosition;
	vec3 mRight;
	vec3 mUp;
	vec3 mProjectionCenter;
};
