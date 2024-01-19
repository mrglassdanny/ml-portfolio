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

            public:
                Layer(ADContext *ctx);

                virtual Tensor *forward(Tensor *x) = 0;
            };
        }
    }
}