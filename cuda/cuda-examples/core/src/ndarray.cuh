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

#define THREADS_PER_BLOCK 32

class Dimensions
{
private:
	std::vector<int> dims_vec_;

public:
	Dimensions();
	Dimensions(int dim_1);
	Dimensions(int dim_1, int dim_2);
	Dimensions(int dim_1, int dim_2, int dim_3);
	Dimensions(int dim_1, int dim_2, int dim_3, int dim_4);
	Dimensions(std::vector<int> dims_vec);
	~Dimensions();

	void print();

	int dim(int dim_idx);
	int count();
	int size();
};

class NdArray
{
protected:
	bool cuda_;
	Dimensions dims_;
	float *data_;

public:
	NdArray(NdArray &src);
	NdArray(bool cuda, Dimensions dims);
	NdArray(bool cuda, int cnt);
	NdArray(bool cuda, int row_cnt, int col_cnt);
	NdArray(bool cuda, int x_cnt, int y_cnt, int z_cnt);
	~NdArray();

	void print();

	bool is_cuda();
	void to_cpu();
	void to_cuda();

	Dimensions dims();
	int num_dims();
	int dims_size();
	int count();
	int rows();
	int cols();
	int xs();
	int ys();
	int zs();
	size_t size();

	float *data();
	void zeros();
	void ones();
	void rands(float mean, float stddev);
};