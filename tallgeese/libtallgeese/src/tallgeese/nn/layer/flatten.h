#pragma once

#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            class Flatten : public Layer
            {
            public:
                Flatten(ADContext *ctx, Shape input_shape);
                ~Flatten();

                virtual Tensor *forward(Tensor *x) override;
            };
        }

    }
}