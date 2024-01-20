#pragma once

#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            class Softmax : public Layer
            {
            public:
                Softmax(ADContext *ctx, Shape input_shape);
                ~Softmax();

                virtual void reset() override;
                virtual Tensor *forward(Tensor *x) override;
            };
        }

    }
}