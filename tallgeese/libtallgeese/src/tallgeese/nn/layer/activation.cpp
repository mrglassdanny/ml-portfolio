#pragma once

#include "activation.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            Activation::Activation(ADContext *ctx, int batch_size, int inputs, ActivationType type)
                : Layer(ctx)
            {
                this->type = type;
                this->y = this->ctx->var(Tensor::zeros({batch_size, inputs}));
            }

            Activation::~Activation()
            {
                delete this->y;
            }

            Tensor *Activation::forward(Tensor *x)
            {
                switch (this->type)
                {
                case Sigmoid:
                    this->y = this->ctx->sigmoid(x, this->y);
                    break;
                case Tanh:
                    break;
                case Relu:
                    break;
                default:
                    break;
                }

                return this->y;
            }
        }
    }
}