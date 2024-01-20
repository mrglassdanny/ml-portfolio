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

                void reset();

            public:
                Layer(ADContext *ctx);

                virtual Tensor *forward(Tensor *x) = 0;

                Shape get_output_shape();
            };
        }
    }
}