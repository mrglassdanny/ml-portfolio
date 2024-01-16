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

namespace epyon
{
    namespace core
    {
        struct IntVar
        {
            float d = 0.0f; // Derivative of context with respect to variable
            float pds[2];   // Partial derivatives of intermediate operation
            IntVar *ps[2];  // Intermediate operation parents

            IntVar();
            IntVar(float pd1, float pd2);
            IntVar(float pd1, float pd2, IntVar *p1, IntVar *p2);
        };

        struct Var
        {
            float v;
            IntVar *iv;

            Var();
            Var(float v);
            Var(float v, IntVar *iv);
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

        class Context;
        class Tensor
        {
        private:
            bool cuda;
            Shape shape;
            float *data;
            bool requires_grad = false;
            IntVar **ivs = nullptr;

            static void print_vec(float *data, int cnt);
            static void print_mtx(float *data, int row_cnt, int col_cnt, const char *whitespace_str);

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

            void require_grad(Context *ctx);

            Shape get_shape();
            int num_dims();
            int count();
            size_t size();

            float sum();
            float min();
            int min_idx();
            float max();
            int max_idx();
            float mean();
            float variance();
            float stddev();

            void abs();

            float get_val(int idx);
            void set_val(int idx, float val);

            float *get_data();
            void zeros();
            void ones();
            void fill(float val);
            void random(float mean, float stddev);
            void random_ints(int upper_bound);
        };

        class Context
        {
        private:
            bool cuda;
            IntVar **tape_blocks;
            int tape_block_cur;
            int tape_iv_cur;

            __host__ __device__ void add_block();

            __host__ __device__ Var op(float v, float pd1, float pd2, IntVar *p1, IntVar *p2);

        public:
            Context(bool cuda);
            ~Context();

            __host__ __device__ IntVar *add_intvar(IntVar iv);

            __host__ __device__ Var var(float v);

            __host__ __device__ void backward();

            __host__ __device__ Var add(Var a, Var b);
            __host__ __device__ Var mul(Var a, Var b);
            __host__ __device__ Var exp(Var a, float b);
        };

    }
}