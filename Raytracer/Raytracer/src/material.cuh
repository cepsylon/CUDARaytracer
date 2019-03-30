#pragma once

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct Material
{
	glm::vec3 mColor;
	glm::vec3 mAmbientAttenuation;
	float mSpecularCoefficient;
	float mShininess;
	float mMagneticPermeability;
	float mRefractionIndex;
};
