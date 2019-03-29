#pragma once

#include "vec3.cuh"

struct Material
{
	vec3 mColor;
	float mSpecularCoefficient;
	float mShininess;
};
