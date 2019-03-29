#include "pointlight.cuh"

__device__ PointLight::PointLight(const vec3 & position, const vec3 & intensity)
	: mPosition(position)
	, mIntensity(intensity)
{ }

__device__ vec3 PointLight::position() const { return mPosition; }
__device__ vec3 PointLight::intensity() const { return mIntensity; }
