#include "camera.cuh"

__device__ Camera::Camera(const vec3 & position, const vec3 & right, const vec3 & up, const vec3 & center)
	: mPosition(position)
	, mRight(right)
	, mUp(up)
	, mProjectionCenter(center)
{ }

__device__ vec3 Camera::position() const { return mPosition; }
__device__ vec3 Camera::right() const { return mRight; }
__device__ vec3 Camera::up() const { return mUp; }
__device__ vec3 Camera::projection_center() const { return mProjectionCenter; }
