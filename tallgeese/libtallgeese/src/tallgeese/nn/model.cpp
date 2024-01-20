#include "model.h"

namespace tallgeese
{
    namespace nn
    {
        Model::Model(LossType loss_type)
        {
            this->ctx = new ADContext();
        }

        Model::Model(LossType loss_type, bool trace)
        {
            this->ctx = new ADContext();
            this->ctx->set_trace(trace);
        }

        Model::~Model()
        {
            delete this->ctx;
            for (auto layer : this->layers)
            {
                delete layer;
            }
        }

        Shape Model::get_output_shape()
        {
            return this->layers[this->layers.size() - 1]->get_output_shape();
        }

        Tensor *Model::forward(Tensor *x)
        {
            auto y = this->ctx->var(x);
            for (int i = 0; i < this->layers.size(); i++)
            {
                y = this->layers[i]->forward(y);
            }
            return y;
        }

        Var Model::loss(Tensor *p, Tensor *y)
        {
            y = this->ctx->var(y);

            switch (this->loss_type)
            {
            case MSE:
                return this->ctx->mse(p, y);
            case CrossEntropy:
                return this->ctx->cross_entropy(p, y);
            default:
                break;
            }

            return Var(0.0f);
        }

        void Model::backward()
        {
            this->ctx->derive();
        }

        void Model::test(Tensor *x, Tensor *y)
        {
            auto p = this->forward(x);
            this->loss(p, y);
            this->backward();

            this->ctx->check_gradients();

            this->ctx->reset();
        }

        void Model::linear(int batch_size, int inputs, int outputs, bool bias)
        {
            this->layers.push_back(new Linear(this->ctx, {batch_size, inputs}, outputs, bias));
        }

        void Model::linear(int batch_size, int inputs, int outputs)
        {
            this->layers.push_back(new Linear(this->ctx, {batch_size, inputs}, outputs, false));
        }

        void Model::linear(int outputs, bool bias)
        {
            this->layers.push_back(new Linear(this->ctx, this->get_output_shape(), outputs, bias));
        }

        void Model::linear(int outputs)
        {
            this->layers.push_back(new Linear(this->ctx, this->get_output_shape(), outputs, false));
        }

        void Model::conv2d(Shape input_shape, Shape filter_shape, bool bias)
        {
            this->layers.push_back(new Conv2d(this->ctx, input_shape, filter_shape, bias));
        }

        void Model::conv2d(Shape input_shape, Shape filter_shape)
        {
            this->layers.push_back(new Conv2d(this->ctx, input_shape, filter_shape, false));
        }

        void Model::conv2d(Shape filter_shape, bool bias)
        {
            this->layers.push_back(new Conv2d(this->ctx, this->get_output_shape(), filter_shape, bias));
        }

        void Model::conv2d(Shape filter_shape)
        {
            this->layers.push_back(new Conv2d(this->ctx, this->get_output_shape(), filter_shape, false));
        }

        void Model::activation(ActivationType type)
        {
            this->layers.push_back(new Activation(this->ctx, this->get_output_shape(), type));
        }

        void Model::softmax()
        {
            this->layers.push_back(new Softmax(this->ctx, this->get_output_shape()));
        }

        void Model::flatten()
        {
            this->layers.push_back(new Flatten(this->ctx, this->get_output_shape()));
        }
    }
}