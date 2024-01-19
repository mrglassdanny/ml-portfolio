#pragma once

#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        class FullyConnectedLayer : public Layer
        {
        private:
            Tensor *w;
            Tensor *b;
            Tensor *z;

        public:
            FullyConnectedLayer(ADContext *ctx);
            FullyConnectedLayer(ADContext *ctx, int batch_size, int inputs, int outputs, bool bias);
            ~FullyConnectedLayer();

            virtual Tensor *forward(Tensor *x);
        };
    }
}