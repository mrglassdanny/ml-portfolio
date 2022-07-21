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

class ArrayNd
{
protected:
	bool cuda_flg;
	Dimensions dims;
	float *data;

	size_t get_data_size();

public:
	ArrayNd(ArrayNd &src);
	ArrayNd(bool cuda_flg, Dimensions dims);
	~ArrayNd();

	virtual void print() = 0;

	bool is_cuda();
	void to_cpu();
	void to_cuda();

	float* get_data();
	void zeros();
	void rands(float mean, float stddev);
};

class Array1d : public ArrayNd
{
public:

	Array1d(bool cuda_flg, int cnt);
	~Array1d();

	virtual void print();

	int get_cnt();
};

class Array2d : public ArrayNd
{
public:

	Array2d(Array2d& src);
	Array2d(bool cuda_flg, int row_cnt, int col_cnt);
	~Array2d();

	virtual void print();

	int get_row_cnt();
	int get_col_cnt();
};

class Array3d : public ArrayNd
{
public:

	Array3d(Array3d& src);
	Array3d(bool cuda_flg, int x, int y, int z);
	~Array3d();

	virtual void print();

	int get_x();
	int get_y();
	int get_z();
};

class Tensor : public ArrayNd
{
public:

	Tensor(Tensor& src);
	Tensor(bool cuda_flg, Dimensions dims);
	~Tensor();

	virtual void print();

	Dimensions get_dims();
	int num_dims();
	int dims_size();
};