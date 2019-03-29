#include "ray.cuh"

__device__ Ray::Ray(const vec3 & point, const vec3 & direction)
	: mPoint(point)
	, mDirection(direction)
{ }

__device__ vec3 Ray::at(float t) const
{
	return mPoint + mDirection * t;
}

__device__ vec3 Ray::point() const { return mPoint; }
__device__ vec3 Ray::direction() const { return mDirection; }
