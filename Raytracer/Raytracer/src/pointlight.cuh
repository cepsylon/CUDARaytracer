#pragma once

#include "vec3.cuh"

#include <cuda_runtime.h>

class PointLight
{
public:
	// Default constructor
	__device__ PointLight() {}
	// Constructor
	__device__ PointLight(const vec3 & position, const vec3 & intensity);

	// Gettors
	__device__ vec3 position() const;
	__device__ vec3 intensity() const;

private:
	vec3 mPosition;
	vec3 mIntensity;
};
