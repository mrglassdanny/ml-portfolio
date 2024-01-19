#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            Layer::Layer(ADContext *ctx)
            {
                this->ctx = ctx;
            }
        }
    }
}