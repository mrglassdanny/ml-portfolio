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

#include "err.cuh"
#include "util.cuh"

#define EPYON_AD_TAPE_BLOCK_SIZE 64
#define EPYON_AD_TAPE_BLOCK_CNT 1024
#define EPYON_AD_INVALID_TAPE_BLOCK -1
#define EPYON_AD_DEFAULT_TAPE_INDEX                              \
    TapeIndex                                                    \
    {                                                            \
        EPYON_AD_INVALID_TAPE_BLOCK, EPYON_AD_INVALID_TAPE_BLOCK \
    }

namespace epyon
{
    namespace core
    {
        struct TapeIndex
        {
            int block = 0;
            int elem = 0;
        };

        struct IntVar
        {
            float d = 0.0f;  // Derivative of context with respect to variable
            float pds[2];    // Partial derivatives of intermediate operation
            TapeIndex ps[2]; // Intermediate operation parents

            __host__ __device__ IntVar();
            __host__ __device__ IntVar(float pd1, float pd2);
            __host__ __device__ IntVar(float pd1, float pd2, TapeIndex p1, TapeIndex p2);
        };

        struct Var
        {
            float v;
            TapeIndex i;

            __host__ __device__ Var();
            __host__ __device__ Var(float v);
            __host__ __device__ Var(float v, TapeIndex i);
        };

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
            Shape(int dim_1, Shape shape);
            ~Shape();

            void print();
            void print_pad(int pad_cnt, bool left_pad_flg);

            int operator[](int) const;
            bool operator==(const Shape &);
            bool operator!=(const Shape &);

            std::vector<int> get_dims();
            int count();
            int size();
        };

        class Tensor
        {
        private:
            bool cuda;
            Shape shape;
            Var *data;

            static void print_vec(Var *data, int cnt);
            static void print_mtx(Var *data, int row_cnt, int col_cnt, const char *whitespace_str);

        public:
            Tensor(Tensor &src);
            Tensor(bool cuda, Shape shape);
            ~Tensor();

            static Tensor *from_data(Shape shape, float *data);
            static Tensor *from_csv(const char *path);
            static void to_csv(const char *path, Tensor *tensor);
            static void to_file(const char *path, Tensor *tensor);

            static Tensor *zeros(bool cuda, Shape shape);
            static Tensor *ones(bool cuda, Shape shape);
            static Tensor *fill(bool cuda, Shape shape, float val);
            static Tensor *random(bool cuda, Shape shape, float mean, float stddev);

            static Tensor *one_hot(Tensor *src);

            void print();

            void copy(Tensor *src);
            void reshape(Shape shape);
            void change_dim(int dim_idx, int dim);

            bool is_cuda();
            void to_cpu();
            void to_cuda();

            Shape get_shape();
            Var *get_data();

            int dims_count();
            int count();
            size_t size();

            float min();
            float max();

            Var get_var(int idx);
            void set_var(int idx, Var var);
            void set_val(int idx, float val);

            void zeros();
            void ones();
            void fill(float val);
            void random(float mean, float stddev);
        };

        class AutoDiffContext
        {
        private:
            bool cuda;
            IntVar **tape;
            int block_cur;
            int elem_cur;

            __host__ __device__ void add_block();
            __host__ __device__ TapeIndex add_intermediate_variable(IntVar iv);

            __host__ __device__ Var op(float v, float pd1, float pd2, TapeIndex p1, TapeIndex p2);

        public:
            AutoDiffContext(bool cuda);
            ~AutoDiffContext();

            __host__ __device__ Var var(float v);
            Tensor *tensor(Tensor *tensor);

            __host__ __device__ void backward();

            __host__ __device__ Var add(Var a, Var b);
            __host__ __device__ Var mul(Var a, Var b);
            __host__ __device__ Var exp(Var a, float b);

            void sum(Tensor *a, Tensor *b);
        };
    }
}