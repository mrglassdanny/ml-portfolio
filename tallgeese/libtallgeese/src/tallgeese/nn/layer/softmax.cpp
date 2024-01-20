#include "softmax.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            Softmax::Softmax(ADContext *ctx, Shape input_shape)
                : Layer(ctx)
            {
                this->y = Tensor::zeros(input_shape);
            }

            Softmax::~Softmax()
            {
                delete this->y;
            }

            void Softmax::reset()
            {
                Layer::reset();
            }

            Tensor *Softmax::forward(Tensor *x)
            {
                this->reset();

                this->y = this->ctx->softmax(x, this->y);

                return this->y;
            }
        }

    }
}