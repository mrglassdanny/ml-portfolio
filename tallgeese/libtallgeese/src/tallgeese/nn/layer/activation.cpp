#pragma once

#include "activation.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            Activation::Activation(ADContext *ctx, Shape input_shape, ActivationType type)
                : Layer(ctx)
            {
                this->type = type;
                this->y = this->ctx->var(Tensor::zeros(input_shape));
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

            Shape Activation::get_output_shape()
            {
                return this->y->shape;
            }
        }
    }
}