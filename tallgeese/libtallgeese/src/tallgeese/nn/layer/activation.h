#pragma once

#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        enum ActivationType
        {
            Sigmoid,
            Tanh,
            Relu
        };

        class ActivationLayer : public Layer
        {
        private:
            ActivationType type;
            Tensor *y;

        public:
            ActivationLayer(ADContext *ctx, int batch_size, int inputs, ActivationType type);
            ~ActivationLayer();

            virtual Tensor *forward(Tensor *x);
        };
    }
}