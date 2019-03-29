#pragma once

#include "surface.cuh"
#include "vector.cuh"

#include <cuda_runtime.h>

class Scene
{
public:
	// Adds surface to scene
	__device__ void add(Surface * surface);

	// Gettor
	__device__ const vector<Surface *> & surfaces() const;
	
private:
	vector<Surface *> mSurfaces;
};
