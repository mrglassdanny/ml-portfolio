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

            void Layer::reset()
            {
                this->y->zeros();
            }

            Shape Layer::get_output_shape()
            {
                return this->y->shape;
            }
        }
    }
}