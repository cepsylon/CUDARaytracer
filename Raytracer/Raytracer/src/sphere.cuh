#pragma once

#include "surface.cuh"
#include "vec3.cuh"

class Sphere
{
public:
	__device__ Sphere(const vec3 & position, float radius);

	__device__ bool collide(const Ray & ray, float t_min, float t_max) const;

private:
	vec3 mPosition;
	float mRadius;
};
