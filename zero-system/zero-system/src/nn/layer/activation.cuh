#pragma once

#include "layer.cuh"

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
            static void evaluate(NdArray *in, int batch_size, int cnt, ActivationType activation);
            static void derive(NdArray *in, NdArray *n, int batch_size, int cnt, ActivationType activation);
            static void summarize(ActivationType activation);
        };
    }
}