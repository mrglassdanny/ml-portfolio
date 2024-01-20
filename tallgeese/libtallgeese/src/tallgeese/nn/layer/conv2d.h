#pragma once

#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            class Conv2d : public Layer
            {
            private:
                Tensor *w;
                Tensor *b;

            public:
                Conv2d(ADContext *ctx, Shape input_shape, Shape filter_shape, bool bias);
                ~Conv2d();

                virtual void reset() override;
                virtual Tensor *forward(Tensor *x) override;
            };
        }
    }
}