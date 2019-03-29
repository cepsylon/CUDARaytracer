#include "scene.cuh"

__device__ void Scene::add(Surface * surface)
{
	mSurfaces.push_back(surface);
}

__device__ const vector<Surface *> & Scene::surfaces() const { return mSurfaces; }
