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
                this->y = nullptr;
            }

            Shape Layer::get_output_shape()
            {
                return this->y->shape;
            }

            void Layer::reset()
            {
                this->y->zeros();
                this->y = this->ctx->var(this->y);
            }
        }
    }
}