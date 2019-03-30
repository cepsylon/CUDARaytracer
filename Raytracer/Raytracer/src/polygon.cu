#include "polygon.cuh"

#include "ray.cuh"

#include <iostream>

__device__ Polygon::Polygon(const Material & material, glm::vec3 * vertices, int count)
	: Surface(material)
	, mVertices(vertices, count)
{
	glm::vec3 v0 = mVertices[1] - mVertices[0];
	glm::vec3 v1 = mVertices[2] - mVertices[0];
	mNormal = glm::normalize(glm::cross(v0, v1));
}

__device__ bool Polygon::collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const
{
	float cos_angle = glm::dot(mNormal, ray.direction());
	if (cos_angle != 0.0f)
	{
		float t = -glm::dot(ray.point() - mVertices.front(), mNormal) / cos_angle;
		if (t_min < t && t < t_max)
		{
			// Get axis to project
			int erase_index = 0;
			if (std::abs(mNormal.x) < std::abs(mNormal.y))
				erase_index = 1;
			if (std::abs(mNormal[erase_index]) < std::abs(mNormal.z))
				erase_index = 2;

			// Project vertices
			glm::vec3 point = ray.at(t);
			vector<glm::vec2> vertices(mVertices.size());
			for (unsigned i = 0; i < vertices.size(); ++i)
			{
				switch (erase_index)
				{
				case 0:
					vertices[i] = glm::vec2{ mVertices[i].y - point.y, mVertices[i].z - point.z };
					break;
				case 1:
					vertices[i] = glm::vec2{ mVertices[i].x - point.x, mVertices[i].z - point.z };
					break;
				case 2:
					vertices[i] = glm::vec2{ mVertices[i].x - point.x, mVertices[i].y - point.y };
					break;
				}
			}

			// Jordan Curve theorem
			int count = 0;
			glm::vec2 start = vertices.back();
			for (int i = 0; i < vertices.size(); ++i)
			{
				const glm::vec2 & vertex = vertices[i];
				glm::vec2 horizontal{ vertex.x, start.x }, vertical{ vertex.y, start.y };
				if (start.x < vertex.x)
				{
					horizontal.x = start.x;
					horizontal.y = vertex.x;
					vertical.x = start.y;
					vertical.y = vertex.y;
				}

				if (horizontal.x < 0.0 && horizontal.y >= 0.0f && (0.0f < vertical.x || 0.0f <= vertical.y))
				{
					float slope = (horizontal.y - horizontal.x) / (vertical.y - vertical.x);
					if (vertical.x - slope * horizontal.x > 0)
						count++;
				}

				start = vertex;
			}

			if (count % 2)
			{
				collision_data.mT = t;
				collision_data.mNormal = mNormal;
				collision_data.mMaterial = mMaterial;
				return true;
			}
		}
	}

	return false;
}

__device__ const vector<glm::vec3> & Polygon::vertices() const { return mVertices; }
