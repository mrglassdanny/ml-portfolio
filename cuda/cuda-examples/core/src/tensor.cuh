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

class Dimensions
{
private:
	std::vector<int> dim_list;

public:
	Dimensions();
	Dimensions(int dim_1);
	Dimensions(int dim_1, int dim_2);
	Dimensions(int dim_1, int dim_2, int dim_3);
	Dimensions(int dim_1, int dim_2, int dim_3, int dim_4);
	Dimensions(std::vector<int> dim_list);
	~Dimensions();

	void print();

	int get_dim(int dim_idx);
	int get_cnt();
	int get_size();
};

class Tensor
{
private:
	bool cuda_flg;
	Dimensions dims;
	float *data;

	size_t get_data_size();

public:
	Tensor(Tensor &src);
	Tensor(bool cuda_flg, Dimensions dims);
	~Tensor();

	void print();

	bool is_cuda();
	Dimensions get_dims();
	int num_dims();
	int dims_size();

	float *get_data();

	void to_cpu();
	void to_cuda();

	void zeros();
	void rands(float mean, float stddev);
};