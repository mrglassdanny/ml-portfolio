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

            public:
                Linear(ADContext *ctx, Shape input_shape, int outputs, bool bias);
                ~Linear();

                virtual Tensor *forward(Tensor *x);
            };
        }
    }
}