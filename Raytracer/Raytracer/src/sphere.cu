#include "sphere.cuh"

#include "ray.cuh"

__device__ Sphere::Sphere(const Material & material, const glm::vec3 & position, float radius)
	: Surface(material)
	, mPosition(position)
	, mRadius(radius)
{ }

__device__ bool Sphere::collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const
{
	// Ray sphere collision
	glm::vec3 to_ray_start = ray.point() - mPosition;
	float a = glm::dot(ray.direction(), ray.direction());
	float b = glm::dot(to_ray_start, ray.direction());
	float c = glm::dot(to_ray_start, to_ray_start) - mRadius * mRadius;

	// Compute discriminant, 4 is gone with the two 2s in b
	float discriminant = b * b - a * c;
	if (0.0f <= discriminant)
	{
		float current_t = (-b - std::sqrt(discriminant)) / a;
		if (t_min <= current_t && current_t <= t_max)
		{
			collision_data.mT = current_t;
			collision_data.mNormal = glm::normalize(ray.at(current_t) - mPosition);
			collision_data.mMaterial = mMaterial;
			return true;
		}

		current_t = (-b + std::sqrt(discriminant)) / a;
		if (t_min <= current_t && current_t <= t_max)
		{
			collision_data.mT = current_t;
			collision_data.mNormal = glm::normalize(ray.at(current_t) - mPosition);
			collision_data.mMaterial = mMaterial;
			return true;
		}
	}

	return false;
}

__device__ glm::vec3 Sphere::position() const { return mPosition; }
__device__ float Sphere::radius() const { return mRadius; }
