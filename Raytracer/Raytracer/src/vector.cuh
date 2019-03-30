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

	__device__ vector(unsigned i)
		: mData(new T[i])
		, mSize(i)
		, mCapacity(i)
	{ }

	__device__ vector(const vector & rhs)
		: mData(new T[rhs.mCapacity])
		, mSize(rhs.mSize)
		, mCapacity(rhs.mCapacity)
	{
		std::memcpy(mData, rhs.mData, sizeof(T) * mSize);
	}

	__device__ vector(vector && rhs)
		: mData(rhs.mData)
		, mSize(rhs.mSize)
		, mCapacity(rhs.mCapacity)
	{
		rhs.mData = nullptr;
		rhs.mSize = 0;
		rhs.mCapacity = 0;
	}

	__device__ vector(T * data, int count)
		: mData(new T[count])
		, mSize(count)
		, mCapacity(count)
	{
		std::memcpy(mData, data, sizeof(T) * mSize);
	}

	__device__ vector & operator=(const vector & rhs)
	{
		if (this != &rhs)
		{
			// Release data
			if (mData)
				delete [] mData;

			// Copy
			mData = new T[rhs.mCapacity];
			mSize = rhs.mSize;
			mCapacity = rhs.mCapacity;
			std::memcpy(mData, rhs.mData, sizeof(T) * mSize);
		}

		return *this;
	}

	__device__ vector & operator=(vector && rhs)
	{
		if (this != rhs)
		{
			// Release data
			if (mData)
				delete [] mData;

			// Move
			mData = rhs.mData; rhs.mData = nullptr;
			mSize = rhs.mSize; rhs.mSize = 0;
			mCapacity = rhs.mCapacity; rhs.mCapacity = 0;
		}

		return *this;
	}

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

	__device__ void pop_back()
	{
		(mData + --mSize)->~T();
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

	__device__ bool empty() const { return mSize == 0; }
	__device__ const T & front() const { return mData[0]; }
	__device__ const T & back() const { return mData[mSize - 1]; }

private:
	__device__ void grow()
	{
		// Duplicate capacity
		if (mCapacity)
			mCapacity *= 2;
		else
			mCapacity = 2;
		T * prev_data = mData;
		mData = new T[mCapacity];
		std::memcpy(mData, prev_data, sizeof(T) * mSize);
		delete [] prev_data;
	}

	T * mData = nullptr;
	int mSize = 0;
	int mCapacity = 2;
};
