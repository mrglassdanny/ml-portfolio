#pragma once

#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            enum ActivationType
            {
                Sigmoid,
                Tanh,
                Relu
            };

            class Activation : public Layer
            {
            private:
                ActivationType type;
                Tensor *y;

            public:
                Activation(ADContext *ctx, int batch_size, int inputs, ActivationType type);
                ~Activation();

                virtual Tensor *forward(Tensor *x);
            };
        }

    }
}