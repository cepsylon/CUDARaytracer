#pragma once

#include "Surface.cuh"
#include "vec3.cuh"
#include "vector.cuh"

class Box : public Surface
{
public:
	__device__ Box(const Material & material, const vec3 & position, const vec3 & width, const vec3 & height, const vec3 & depth);

	__device__ bool collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const override;

	struct Plane
	{
		vec3 mCenter;
		vec3 mNormal;
	};

private:
	vector<Plane> mPlanes;
};
