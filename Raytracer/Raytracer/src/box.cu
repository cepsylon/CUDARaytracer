#include "box.cuh"

#include "ray.cuh"

__device__ Box::Box(const Material & material, const glm::vec3 & position, const glm::vec3 & width, const glm::vec3 & height, const glm::vec3 & depth)
	: Surface(material)
	, mPlanes(6)
{
	glm::vec3 normals[] = { glm::normalize(glm::cross(width, height)), glm::normalize(glm::cross(height, depth)), glm::normalize(glm::cross(depth, width)) };
	mPlanes[0] = Plane{ position, normals[0] };
	mPlanes[1] = Plane{ position + depth, -normals[0] };
	mPlanes[2] = Plane{ position, normals[1] };
	mPlanes[3] = Plane{ position + width, -normals[1] };
	mPlanes[4] = Plane{ position, normals[2] };
	mPlanes[5] = Plane{ position + height, -normals[2] };
}

__device__ bool Box::collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const
{
	int min_index = -1;
	int max_index = -1;
	float min = t_min;
	for (unsigned i = 0; i < mPlanes.size() && t_min <= t_max; ++i)
	{
		const Plane & plane = mPlanes[i];
		glm::vec3 to_ray = ray.point() - plane.mCenter;
		const glm::vec3 & normal = plane.mNormal;
		float cos_angle = glm::dot(normal, ray.direction());

		if (cos_angle < 0.0f)
		{
			float value = -glm::dot(to_ray, normal) / cos_angle;
			if (value > t_min)
			{
				t_min = value;
				min_index = i;
			}
		}
		else if (cos_angle > 0.0f)
		{
			float value = -glm::dot(to_ray, normal) / cos_angle;
			if (value < t_max)
			{
				t_max = value;
				max_index = i;
			}
		}
		// No collision
		else if (glm::dot(to_ray, normal) > 0.0f)
			t_min = t_max + 1;
	}

	// Check if we collided
	if (t_min <= t_max)
	{
		if (t_min != min)
		{
			collision_data.mNormal = mPlanes[min_index].mNormal;
			collision_data.mT = t_min;
		}
		else
		{
			collision_data.mNormal = mPlanes[max_index].mNormal;
			collision_data.mT = t_max;
		}
		collision_data.mMaterial = mMaterial;
	}

	return t_min <= t_max;
}
