#pragma once

#include <stdio.h>
#include <stdlib.h>
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
        private:
            std::vector<int> shape;
            Var *data;

        public:
            Tensor(std::vector<int> shape);
            Tensor(std::vector<int> shape, float val);
            Tensor(std::vector<int> shape, float mean, float stddev);
            ~Tensor();

            static Tensor *zeros(std::vector<int> shape);
            static Tensor *fill(std::vector<int> shape, float val);
            static Tensor *random(std::vector<int> shape);
            static Tensor *random(std::vector<int> shape, float mean, float stddev);

            void print();

            std::vector<int> get_shape();
            Var *get_data();

            bool has_same_shape(Tensor *other);

            int count();
            int size();

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

            void validate_shapes(Tensor *a, Tensor *b);

        public:
            ADContext();
            ADContext(bool trace);

            Var var(float v);
            Tensor *var(Tensor *tensor);
            Var parm(float v);
            Tensor *parm(Tensor *tensor);

            void derive();
            float get_derivative(Var var);

            void reset_gradients();
            void check_gradients();

            Var add(Var a, Var b);
            Var multiply(Var a, Var b);
            Var power(Var a, Var b);
            Var sigmoid(Var a);

            Var dot(Tensor *a, Tensor *b);
            Tensor *dot(Tensor *a, Tensor *b, Tensor *c);

            Tensor *sigmoid(Tensor *a, Tensor *b);
        };
    }
}