#include "ray.cuh"

__device__ Ray::Ray(const glm::vec3 & point, const glm::vec3 & direction)
	: mPoint(point)
	, mDirection(direction)
{ }

__device__ glm::vec3 Ray::at(float t) const
{
	return mPoint + mDirection * t;
}

__device__ glm::vec3 Ray::point() const { return mPoint; }
__device__ glm::vec3 Ray::direction() const { return mDirection; }
