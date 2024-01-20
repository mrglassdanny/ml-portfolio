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

            public:
                Activation(ADContext *ctx, Shape input_shape, ActivationType type);
                ~Activation();

                virtual void reset() override;
                virtual Tensor *forward(Tensor *x) override;
            };
        }

    }
}