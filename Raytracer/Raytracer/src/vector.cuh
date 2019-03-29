#pragma once

#include <cuda_runtime.h>

#include <cstring>

template <typename T>
class vector
{
public:
	__device__ vector()
	: mData(new T[2u])
	, mSize(0u)
	, mCapacity(2u)
	{	}

	__device__ ~vector()
	{
		if (mData)
			delete[] mData;
	}

	__device__ void push_back(const T & value)
	{
		// Resize vector
		if (mSize == mCapacity)
			grow();

		mData[mSize++] = value;
	}

	__device__ unsigned size() const
	{
		return mSize;
	}

	__device__ const T & operator[](unsigned index) const
	{
		// No throws in kernels, we would need to create a wrapper for that
		if (mSize <= index)
			printf("Index out of range\n");

		return mData[index];
	}

	__device__ T & operator[](unsigned index)
	{
		// No throws in kernels, we would need to create a wrapper for that
		if (mSize <= index)
			printf("Index out of range\n");

		return mData[index];
	}

private:
	__device__ void grow()
	{
		// Duplicate capacity
		mCapacity *= 2;
		T * prev_data = mData;
		mData = new T[mCapacity];
		std::memcpy(mData, prev_data, sizeof(T) * mSize);
		delete [] prev_data;
	}

	T * mData = nullptr;
	int mSize = 0;
	int mCapacity = 2;
};
