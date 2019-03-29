#pragma once

#include "surface.cuh"
#include "vector.cuh"
#include "camera.cuh"

#include <cuda_runtime.h>

class Scene
{
public:
	// Adds surface to scene
	__device__ void add(Surface * surface);

	// Set camera
	__device__ void set_camera(const Camera & camera);

	// Gettors
	__device__ const vector<Surface *> & surfaces() const;
	__device__ const Camera & camera() const;
	
private:
	Camera mCamera;
	vector<Surface *> mSurfaces;
};
