#include "scene.cuh"

__device__ void Scene::add(Surface * surface)
{
	mSurfaces.push_back(surface);
}

__device__ void Scene::set_camera(const Camera & camera) { mCamera = camera; }

__device__ const vector<Surface *> & Scene::surfaces() const { return mSurfaces; }
__device__ const Camera & Scene::camera() const { return mCamera; }
