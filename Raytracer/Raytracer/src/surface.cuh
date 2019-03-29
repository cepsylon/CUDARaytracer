#pragma once

#include "vec3.cuh"
#include "material.cuh"

#include <cuda_runtime.h>

#include <cfloat> // FLT_MAX

class Ray;

struct CollisionData
{
	vec3 mNormal{ 0.0f };
	float mT = FLT_MAX;
	Material mMaterial;
};

class Surface
{
public:
	__device__ Surface(const Material & material);
	__device__ virtual bool collide(const Ray & ray, float t_min, float t_max, CollisionData & collision_data) const = 0;

protected:
	Material mMaterial;
};
