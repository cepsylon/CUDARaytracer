#pragma once

#include <cuda_runtime.h>

#include <cmath>

struct vec3
{
	// Default constructor
	__host__ __device__ vec3();
	// Constructor to assign value to xyz
	__host__ __device__ vec3(float value);
	// Custom constructor
	__host__ __device__ vec3(float x, float y, float z);

	// Operator overloads
	__host__ __device__ vec3 operator+(const vec3 & rhs) const;
	__host__ __device__ vec3 & operator+=(const vec3 & rhs);
	__host__ __device__ vec3 operator-(const vec3 & rhs) const;
	__host__ __device__ vec3 & operator-=(const vec3 & rhs);
	__host__ __device__ vec3 operator-() const;
	__host__ __device__ vec3 operator*(float value) const;
	__host__ __device__ vec3 & operator*=(float value);
	__host__ __device__ vec3 operator/(float value) const;
	__host__ __device__ vec3 & operator/=(float value);

	// Dot product
	__host__ __device__ static float dot(const vec3 & a, const vec3 & b);

	// Computes length of vector
	__host__ __device__ static float length(const vec3 & value);

	// Normalizes vector
	__host__ __device__ static vec3 normalize(const vec3 & value);

	// Cross product
	__host__ __device__ static vec3 cross(const vec3 & a, const vec3 & b);

	union
	{
		struct
		{
			float x, y, z;
		};
		struct
		{
			float r, g, b;
		};
	};
};
