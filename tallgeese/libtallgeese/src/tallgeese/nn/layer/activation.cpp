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
                this->y = Tensor::zeros(input_shape);
            }

            Activation::~Activation()
            {
                delete this->y;
            }

            Tensor *Activation::forward(Tensor *x)
            {
                this->reset();

                this->y = this->ctx->var(this->y);

                switch (this->type)
                {
                case Sigmoid:
                    this->y = this->ctx->sigmoid(x, this->y);
                    break;
                case Tanh:
                    this->y = this->ctx->tanh(x, this->y);
                    break;
                case Relu:
                    this->y = this->ctx->relu(x, this->y);
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