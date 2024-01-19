#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <random>

#include "err.h"
#include "constants.h"

namespace tallgeese
{
    namespace core
    {
        typedef std::vector<int> Shape;

        enum Operation
        {
            None = 0,
            Parameter,
            Add,
            Multiply,
            Power,
            Sigmoid
        };

        struct Var
        {
            float v = 0.0f;
            int i = TALLGEESE_CORE_INVALID_INTVAR_INDEX;

            Var();
            Var(float v);
            Var(float v, int i);

            void print();
        };

        struct IntVar
        {
            Operation op = None;
            float d = 0.0f;
            float pds[2] = {0.0f, 0.0f};
            int is[2] = {TALLGEESE_CORE_INVALID_INTVAR_INDEX, TALLGEESE_CORE_INVALID_INTVAR_INDEX};

            IntVar();
            IntVar(Operation op);
            IntVar(Operation op, float pd1, float pd2);
            IntVar(Operation op, float pd1, float pd2, int i1, int i2);

            void print();
        };

        class Tensor
        {

        public:
            Shape shape;
            Var *data;

            Tensor(Shape shape);
            Tensor(Shape shape, float val);
            Tensor(Shape shape, float mean, float stddev);
            ~Tensor();

            static Tensor *zeros(Shape shape);
            static Tensor *fill(Shape shape, float val);
            static Tensor *random(Shape shape);
            static Tensor *random(Shape shape, float mean, float stddev);

            Var get_var(int dims, ...);
            void set_var(Var var, int dims, ...);

            void print();

            bool has_same_shape(Tensor *other);

            int dims();
            int count();
            size_t size();

            void zeros();
            void fill(float val);
            void random();
            void random(float mean, float stddev);
        };

        class ADContext
        {
        private:
            std::vector<IntVar> tape;

            bool trace = false;
            std::vector<Var> vars;
            bool replaying = false;

            Var op(Operation op, float v, float pd1, float pd2, int i1, int i2);

            Var evaluate();

            void validate_shapes_are_same(Tensor *a, Tensor *b);

        public:
            ADContext();
            ADContext(bool trace);

            Var var(float v);
            Tensor *var(Tensor *tensor);
            Var parm(float v);
            Tensor *parm(Tensor *tensor);

            void reset();

            void derive();
            float get_derivative(Var var);

            void check_gradients();

            Var add(Var a, Var b);
            Var multiply(Var a, Var b);
            Var power(Var a, Var b);
            Var sigmoid(Var a);

            Var dot(Tensor *a, Tensor *b);
            Tensor *dot(Tensor *a, Tensor *b, Tensor *c);

            Tensor *matrix_multiply(Tensor *x, Tensor *w, Tensor *y);

            Tensor *sigmoid(Tensor *a, Tensor *b);
        };
    }
}