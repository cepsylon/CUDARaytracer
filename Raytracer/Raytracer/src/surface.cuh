#pragma once

#include <cuda_runtime.h>

class Ray;

class Surface
{
public:
	__device__ virtual bool collide(const Ray & ray, float t_min, float t_max) const = 0;
};
