#pragma once

#include "../../core/mod.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        namespace layer
        {
            enum ActivationType
            {
                None,
                Sigmoid,
                Tanh,
                ReLU
            };

            class Activation
            {
            public:
                static void evaluate(Tensor *in, int batch_size, int cnt, ActivationType activation);
                static void derive(Tensor *in, Tensor *n, int batch_size, int cnt, ActivationType activation);
                static void summarize(ActivationType activation);
            };
        }
    }
}