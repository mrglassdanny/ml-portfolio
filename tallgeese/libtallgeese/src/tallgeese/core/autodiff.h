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

        enum OpType
        {
            Variable = 0,
            Parameter,
            Add,
            Multiply,
            Power,
            Exponential,
            NaturalLog,
            Sine,
            Cosine,
            Sigmoid,
            Tanh,
            Relu
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
            OpType op = Variable;
            float d = 0.0f;
            float pds[2] = {0.0f, 0.0f};
            int is[2] = {TALLGEESE_CORE_INVALID_INTVAR_INDEX, TALLGEESE_CORE_INVALID_INTVAR_INDEX};

            IntVar();
            IntVar(OpType op);
            IntVar(OpType op, float pd1, float pd2);
            IntVar(OpType op, float pd1, float pd2, int i1, int i2);

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
            static Tensor *from_data(Shape shape, float *data);
            static Tensor *one_hot(Tensor *src, int max);

            Var get_var(...);
            void set_var(Var var, ...);

            void copy_data(Tensor *other);

            void print();

            bool has_same_shape(Tensor *other);

            int dims();
            int count();
            size_t size();

            float min();
            float max();

            void zeros();
            void fill(float val);
            void random();
            void random(float mean, float stddev);
        };

        class ADContext
        {
        private:
            bool trace = false;
            std::vector<Var> vars;
            bool replaying = false;

            Var op(OpType op, float v, float pd1, float pd2, int i1, int i2);

            Var evaluate();

            void validate_shapes_are_same(Tensor *a, Tensor *b);

        public:
            std::vector<IntVar> tape;
            std::vector<Var *> parms;

            ADContext();

            Var var(float v);
            Tensor *var(Tensor *tensor);
            Var parm(float v);
            Tensor *parm(Tensor *tensor);

            void set_trace(bool on);

            void reset();

            void derive();
            float get_derivative(Var var);

            void check_gradients();

            Var negative(Var a);
            Var add(Var a, Var b);
            Var subtract(Var a, Var b);
            Var multiply(Var a, Var b);
            Var divide(Var a, Var b);
            Var power(Var a, Var b);
            Var exponential(Var a);
            Var natural_log(Var a);
            Var sine(Var a);
            Var cosine(Var a);
            Var sigmoid(Var a);
            Var tanh(Var a);
            Var relu(Var a);
            Var mse(Var p, Var y);

            Var dot(Tensor *a, Tensor *b);
            Tensor *dot(Tensor *a, Tensor *b, Tensor *c);

            Tensor *matrix_multiply(Tensor *x, Tensor *w, Tensor *y);

            Tensor *sigmoid(Tensor *a, Tensor *b);
            Tensor *tanh(Tensor *a, Tensor *b);
            Tensor *relu(Tensor *a, Tensor *b);
            Var mse(Tensor *p, Tensor *y);
            Tensor *softmax(Tensor *x, Tensor *y);
            Var cross_entropy(Tensor *p, Tensor *y);
        };
    }
}