#pragma once

#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            class Linear : public Layer
            {
            private:
                Tensor *w;
                Tensor *b;
                Tensor *y;

            public:
                Linear(ADContext *ctx, int batch_size, int inputs, int outputs, bool bias);
                ~Linear();

                virtual Tensor *forward(Tensor *x);
            };
        }
    }
}