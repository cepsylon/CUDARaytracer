#pragma once

#include "vec3.cuh"

class Ray
{
public:
	// Constructor
	__device__ Ray(const vec3 & point, const vec3 & direction);

	// Compute point at t
	__device__ vec3 at(float t) const;

	// Gettors
	__device__ vec3 point() const;
	__device__ vec3 direction() const;
private:
	vec3 mPoint;
	vec3 mDirection;
};
