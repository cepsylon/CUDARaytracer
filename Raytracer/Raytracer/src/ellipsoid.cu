#include "ellipsoid.cuh"

#include "ray.cuh"

__device__ Ellipsoid::Ellipsoid(const Material & material, const glm::vec3 & position, const glm::vec3 & width, const glm::vec3 & height, const glm::vec3 & depth)
	: Surface(material)
	, mPosition(position)
	, mWidth(width)
	, mHeight(height)
	, mDepth(depth)
{ }

__device__ bool Ellipsoid::collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const
{
	// Convert ray to ellipsoid space
	glm::mat3 transform_matrix = glm::inverse(glm::mat3{ mWidth, mHeight, mDepth });
	Ray transformed_ray{ transform_matrix * (ray.point() - mPosition), transform_matrix * ray.direction() };

	// Sphere ray intersection
	float a = glm::dot(transformed_ray.direction(), transformed_ray.direction());
	float b = glm::dot(transformed_ray.point(), transformed_ray.direction());
	float c = glm::dot(transformed_ray.point(), transformed_ray.point()) - 1.0f;

	// Compute discriminant, 4 is gone with the two 2s in b
	float discriminant = b * b - a * c;
	if (0.0f <= discriminant)
	{
		float current_t = (-b - std::sqrt(discriminant)) / a;
		if (t_min <= current_t && current_t <= t_max)
		{
			glm::vec3 normal = ray.at(current_t) - mPosition;
			normal = glm::transpose(transform_matrix) * transform_matrix * normal;
			collision_data.mNormal = glm::normalize(normal);
			collision_data.mT = current_t;
			collision_data.mMaterial = mMaterial;
			return true;
		}

		current_t = (-b + std::sqrt(discriminant)) / a;
		if (t_min <= current_t && current_t <= t_max)
		{
			glm::vec3 normal = ray.at(current_t) - mPosition;
			normal = glm::transpose(transform_matrix) * transform_matrix * normal;
			collision_data.mNormal = glm::normalize(normal);
			collision_data.mT = current_t;
			collision_data.mMaterial = mMaterial;
			return true;
		}
	}

	return false;
}
