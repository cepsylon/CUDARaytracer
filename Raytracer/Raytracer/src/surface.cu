#include "surface.cuh"

__device__ Surface::Surface(const vec3 & color)
	: mColor(color)
{ }
