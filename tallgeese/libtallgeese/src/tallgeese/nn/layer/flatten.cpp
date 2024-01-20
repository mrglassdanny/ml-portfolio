
#include "flatten.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            Flatten::Flatten(ADContext *ctx, Shape input_shape)
                : Layer(ctx)
            {
                int inputs = 1;
                for (int i = 1; i < input_shape.size(); i++)
                {
                    inputs *= input_shape[i];
                }

                this->y = this->ctx->var(Tensor::zeros({input_shape[0], inputs}));
            }

            Flatten::~Flatten()
            {
                delete this->y;
            }

            Tensor *Flatten::forward(Tensor *x)
            {
                this->y->copy_data(x);
                return this->y;
            }

            Shape Flatten::get_output_shape()
            {
                return this->y->shape;
            }
        }

    }
}