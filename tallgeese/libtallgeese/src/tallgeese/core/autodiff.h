#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <random>

#define TALLGEESE_CORE_INVALID_INTVAR_INDEX -1

namespace tallgeese
{
    namespace core
    {
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
            float d = 0.0f;
            float pds[2] = {0.0f, 0.0f};
            int is[2] = {TALLGEESE_CORE_INVALID_INTVAR_INDEX, TALLGEESE_CORE_INVALID_INTVAR_INDEX};

            IntVar();
            IntVar(float pd1, float pd2);
            IntVar(float pd1, float pd2, int i1, int i2);
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

            void print();

            Var *get_data();

            int count();
            int size();

            void zeros();
            void fill(float val);
            void random(float mean, float stddev);
        };

        class ADContext
        {
        private:
            std::vector<IntVar> tape;

            Var op(float v, float pd1, float pd2, int i1, int i2);

        public:
            ADContext();

            Var var(float v);
            Tensor *tensor(Tensor *tensor);

            void derive();
            float get_derivative(Var var);

            void check_grad();

            Var add(Var a, Var b);
            Var mul(Var a, Var b);
            Var pwr(Var a, float b);

            void sum(Tensor *a, Tensor *b);
        };
    }
}