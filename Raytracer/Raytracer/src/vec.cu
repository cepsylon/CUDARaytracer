#include "vec3.cuh"

__host__ __device__ vec3::vec3()
	: x(0.0f)
	, y(0.0f)
	, z(0.0f)
{ }

__host__ __device__ vec3::vec3(float value)
	: x(value)
	, y(value)
	, z(value)
{ }

__host__ __device__ vec3::vec3(float x, float y, float z)
	: x(x)
	, y(y)
	, z(z)
{ }


__host__ __device__ inline vec3 vec3::operator+(const vec3 & rhs) const
{
	return vec3{ x + rhs.x, y + rhs.y, z + rhs.z };
}

__host__ __device__ inline vec3 & vec3::operator+=(const vec3 & rhs)
{
	x += rhs.x; y += rhs.y; z += rhs.z;
	return *this;
}

__host__ __device__ inline vec3 vec3::operator-(const vec3 & rhs) const
{
	return vec3{ x - rhs.x, y - rhs.y, z - rhs.z };
}

__host__ __device__ inline vec3 & vec3::operator-=(const vec3 & rhs)
{
	x -= rhs.x; y -= rhs.y; z -= rhs.z;
	return *this;
}

__host__ __device__ inline vec3 vec3::operator-() const
{
	return vec3{ -x, -y, -z };
}

__host__ __device__ inline vec3 vec3::operator*(float value) const
{
	return vec3{ x * value, y * value, z * value };
}

__host__ __device__ inline vec3 & vec3::operator*=(float value)
{
	x *= value; y *= value; z *= value;
	return *this;
}

__host__ __device__ inline vec3 vec3::operator/(float value) const
{
	return vec3{ x / value, y / value, z / value };
}

__host__ __device__ inline vec3 & vec3::operator/=(float value)
{
	x /= value; y /= value; z /= value;
	return *this;
}

__host__ __device__ float vec3::dot(const vec3 & a, const vec3 & b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ float vec3::length(const vec3 & value)
{
	return std::sqrt(dot(value, value));
}

__host__ __device__ vec3 vec3::normalize(const vec3 & value)
{
	return value / value.length(value);
}

__host__ __device__ vec3 vec3::cross(const vec3 & a, const vec3 & b)
{
	return vec3{ a.y * b.z - b.y * a.z,
							 a.z * b.x - b.z * a.x,
							 a.x * b.y - b.x * a.y };
}
