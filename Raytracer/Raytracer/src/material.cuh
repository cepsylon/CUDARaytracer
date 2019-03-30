#pragma once

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct Material
{
	glm::vec3 mColor;
	float mSpecularCoefficient;
	float mShininess;
};
