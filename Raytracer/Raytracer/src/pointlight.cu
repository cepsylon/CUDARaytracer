#include "pointlight.cuh"

__device__ PointLight::PointLight(const glm::vec3 & position, const glm::vec3 & intensity, float radius)
	: mPosition(position)
	, mIntensity(intensity)
	, mRadius(radius)
{ }

__device__ glm::vec3 PointLight::rand_pos(curandState * random_state) const
{
	glm::vec3 random;
	float length = 0.0f;
	do
	{
		random.x = curand_uniform(random_state) * 2.0f - 1.0f;
		random.y = curand_uniform(random_state) * 2.0f - 1.0f;
		random.z = curand_uniform(random_state) * 2.0f - 1.0f;

		length = glm::dot(random, random);
	} while (length > 1.0f);

	return random * mRadius + mPosition;
}

__device__ glm::vec3 PointLight::position() const { return mPosition; }
__device__ glm::vec3 PointLight::intensity() const { return mIntensity; }
