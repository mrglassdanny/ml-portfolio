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

class ArrayNd
{
protected:
	bool cuda_;
	Dimensions dims_;
	float *data_;

public:
	ArrayNd(ArrayNd &src);
	ArrayNd(bool cuda, Dimensions dims);
	~ArrayNd();

	void print();

	bool is_cuda();
	void to_cpu();
	void to_cuda();

	Dimensions dims();
	int num_dims();
	int dims_size();

	float *data();
	int count();
	size_t size();

	void zeros();
	void ones();
	void rands(float mean, float stddev);
};

class Array1d : public ArrayNd
{
public:
	Array1d(bool cuda, int cnt);
	~Array1d();

	void print();
};

class Array2d : public ArrayNd
{
public:
	Array2d(bool cuda, int row_cnt, int col_cnt);
	~Array2d();

	void print();

	int rows();
	int cols();
};

class Array3d : public ArrayNd
{
public:
	Array3d(bool cuda, int x_cnt, int y_cnt, int z_cnt);
	~Array3d();

	void print();

	int xs();
	int ys();
	int zs();
};