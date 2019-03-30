#include "scene.cuh"

__device__ Scene::~Scene()
{
	for (int i = 0; i < mSurfaces.size(); ++i)
		delete mSurfaces[i];
}

__device__ void Scene::add(Surface * surface)
{
	mSurfaces.push_back(surface);
}

__device__ void Scene::add(const PointLight & light)
{
	mLights.push_back(light);
}

__device__ void Scene::set_camera(const Camera & camera) { mCamera = camera; }
__device__ void Scene::set_ambient(const vec3 & ambient) { mAmbient = ambient; }

__device__ const Camera & Scene::camera() const { return mCamera; }
__device__ const vector<Surface *> & Scene::surfaces() const { return mSurfaces; }
__device__ const vector<PointLight> & Scene::lights() const { return mLights; }
__device__ const vec3 & Scene::ambient() const { return mAmbient; }
