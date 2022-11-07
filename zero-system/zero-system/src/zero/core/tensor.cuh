#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "util.cuh"
#include "err.cuh"

namespace zero
{
	namespace core
	{
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
			void print_pad(int pad_cnt, bool left_pad_flg);

			int operator[](int) const;
			bool operator==(const Shape &);
			bool operator!=(const Shape &);

			std::vector<int> dims();
			int num_dims();
			int dims_size();
		};

		class Tensor
		{
		private:
			bool cuda_;
			Shape shape_;
			float *data_;

			static void print_vec(float *data, int cnt);
			static void print_mtx(float *data, int row_cnt, int col_cnt, const char *whitespace_str);

		public:
			Tensor(Tensor &src);
			Tensor(bool cuda, Shape shape);
			~Tensor();

			static Tensor *from_data(Shape shape, float *data);
			static Tensor *from_csv(const char *path);
			static void to_csv(const char *path, Tensor *arr);
			static void to_file(const char *path, Tensor *arr);

			static Tensor *zeros(bool cuda, Shape shape);
			static Tensor *ones(bool cuda, Shape shape);
			static Tensor *full(bool cuda, Shape shape, float val);
			static Tensor *random(bool cuda, Shape shape, float mean, float stddev);
			static Tensor *random_ints(bool cuda, Shape shape, int upper_bound);

			static Tensor *one_hot(Tensor *src, int max_val);
			static Tensor *pad(Tensor *src, int pad_row_cnt, int pad_col_cnt);
			static Tensor *unpad(Tensor *src, int pad_row_cnt, int pad_col_cnt);

			void print();

			void copy(Tensor *src);
			void reshape(Shape shape);
			void change_dim(int dim_idx, int dim);

			bool is_cuda();
			void to_cpu();
			void to_cuda();

			Shape shape();
			int num_dims();
			int dims_size();
			size_t size();

			int count();
			float sum();
			float min();
			float max();
			float mean();
			float variance();
			float stddev();

			float get_val(int idx);
			void set_val(int idx, float val);

			float *data();
			void zeros();
			void ones();
			void full(float val);
			void random(float mean, float stddev);
			void random_ints(int upper_bound);
		};
	}
}