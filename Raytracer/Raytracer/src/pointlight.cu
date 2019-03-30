#include "pointlight.cuh"

__device__ PointLight::PointLight(const glm::vec3 & position, const glm::vec3 & intensity)
	: mPosition(position)
	, mIntensity(intensity)
{ }

__device__ glm::vec3 PointLight::position() const { return mPosition; }
__device__ glm::vec3 PointLight::intensity() const { return mIntensity; }
