#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#define THREADS_PER_BLOCK 32

class Shape
{
private:
	std::vector<int> dims_;

public:
	Shape();
	Shape(int dim_1);
	Shape(int dim_1, int dim_2);
	Shape(int dim_1, int dim_2, int dim_3);
	Shape(int dim_1, int dim_2, int dim_3, int dim_4);
	Shape(std::vector<int> dims);
	Shape(int dim_1, Shape shape);
	~Shape();

	void print();

	int operator[](int) const;
	std::vector<int> dims();
	int num_dims();
	int dims_size();
};

class NdArray
{
protected:
	bool cuda_;
	Shape shape_;
	float *data_;

public:
	NdArray(NdArray &src);
	NdArray(bool cuda, Shape shape);
	~NdArray();

	static NdArray *zeros(bool cuda, Shape shape);
	static NdArray *ones(bool cuda, Shape shape);
	static NdArray *rands(bool cuda, Shape shape, float mean, float stddev);

	void print();
	void copy(NdArray *src);
	void reshape(Shape shape);
	void change_dim(int dim_idx, int dim);

	bool is_cuda();
	void to_cpu();
	void to_cuda();

	Shape shape();
	int num_dims();
	int dims_size();

	int count();
	size_t size();

	float *data();
	void zeros();
	void ones();
	void rands(float mean, float stddev);

	float get_val(int idx);
	void set_val(int idx, float val);

	float sum();
	float min();
	float max();
	float mean();
	float stddev();
};