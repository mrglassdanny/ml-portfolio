#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

class Shape
{
private:
	std::vector<int> dims;

public:
	Shape();
	Shape(int dim_1);
	Shape(int dim_1, int dim_2);
	Shape(int dim_1, int dim_2, int dim_3);
	Shape(int dim_1, int dim_2, int dim_3, int dim_4);
	Shape(std::vector<int> dims);
	~Shape();

	void print();

	int get_dim(int dim_idx);
	int get_dims_cnt();
	int get_dims_size();
};

class Tensor
{
private:
	bool cuda_flg;
	Shape shape;
	float *data;

public:
	Tensor(Tensor &src);
	Tensor(bool cuda_flg, Shape shape);
	~Tensor();

	void print();

	Shape get_shape();

	void to_cpu();
	void to_cuda();

	void zeros();
	void rands(float mean, float stddev);
};