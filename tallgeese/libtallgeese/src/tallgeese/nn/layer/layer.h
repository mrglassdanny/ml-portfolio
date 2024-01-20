#pragma once

#include "../../core/mod.h"

using namespace tallgeese::core;

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            class Layer
            {
            protected:
                ADContext *ctx;
                Tensor *y;

            public:
                Layer(ADContext *ctx);

                virtual void reset();
                virtual Tensor *forward(Tensor *x) = 0;

                Shape get_output_shape();
            };
        }
    }
}