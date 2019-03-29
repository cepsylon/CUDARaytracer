#pragma once

#include "surface.cuh"
#include "vec3.cuh"

class Sphere : public Surface
{
public:
	__device__ Sphere(const vec3 & color, const vec3 & position, float radius);

	__device__ bool collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const override;

private:
	vec3 mPosition;
	float mRadius;
};
