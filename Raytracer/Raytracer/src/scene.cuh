#pragma once

#include "surface.cuh"
#include "pointlight.cuh"
#include "vector.cuh"
#include "camera.cuh"

#include <cuda_runtime.h>

class Scene
{
public:
	// Destructor to free memory
	__device__ ~Scene();

	// Adds surface to scene
	__device__ void add(Surface * surface);
	// Adds light to scene
	__device__ void add(const PointLight & light);

	// Set camera
	__device__ void set_camera(const Camera & camera);

	// Gettors
	__device__ const Camera & camera() const;
	__device__ const vector<Surface *> & surfaces() const;
	__device__ const vector<PointLight> & lights() const;
	
private:
	Camera mCamera;
	vector<Surface *> mSurfaces;
	vector<PointLight> mLights;
};
