#include "surface.cuh"

__device__ Surface::Surface(const Material & material)
	: mMaterial(material)
{ }
