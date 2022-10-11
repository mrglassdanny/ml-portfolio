#pragma once

#include "layer.cuh"

namespace nn
{
    namespace layer
    {
        enum Activation
        {
            None,
            Sigmoid,
            Tanh,
            ReLU
        };

        void summarize_activation(Activation activation);
    }
}