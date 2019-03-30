#include "camera.cuh"

__device__ Camera::Camera(const glm::vec3 & position, const glm::vec3 & right, const glm::vec3 & up, const glm::vec3 & center)
	: mPosition(position)
	, mRight(right)
	, mUp(up)
	, mProjectionCenter(center)
{ }

__device__ glm::vec3 Camera::position() const { return mPosition; }
__device__ glm::vec3 Camera::right() const { return mRight; }
__device__ glm::vec3 Camera::up() const { return mUp; }
__device__ glm::vec3 Camera::projection_center() const { return mProjectionCenter; }
