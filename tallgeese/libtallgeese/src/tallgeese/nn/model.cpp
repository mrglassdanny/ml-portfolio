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

        int Model::get_batch_size()
        {
            return this->get_output_shape()[0];
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

        float Model::accuracy(Tensor *p, Tensor *y, int (*acc_fn)(Tensor *p, Tensor *y, int batch_size))
        {
            int batch_size = this->get_batch_size();

            int correct_cnt = acc_fn(p, y, batch_size);

            return ((float)correct_cnt / (float)batch_size);
        }

        void Model::backward()
        {
            this->ctx->derive();
        }

        void Model::step(float lr)
        {
            for (auto parm : this->ctx->parms)
            {
                parm->v -= lr * this->ctx->tape[parm->i].d / this->get_batch_size();
            }
        }

        void Model::reset()
        {
            this->ctx->reset();
        }

        void Model::test(Tensor *x, Tensor *y)
        {
            auto p = this->forward(x);
            this->loss(p, y);
            this->backward();

            this->ctx->check_gradients();

            this->reset();
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

        void Model::flatten(Shape input_shape)
        {
            this->layers.push_back(new Flatten(this->ctx, input_shape));
        }

        void Model::flatten()
        {
            this->layers.push_back(new Flatten(this->ctx, this->get_output_shape()));
        }

        int Model::regression_accuracy_fn(Tensor *p, Tensor *y, int batch_size)
        {
            int correct_cnt = 0;

            for (int i = 0; i < batch_size; i++)
            {
                if (p->data[i].v == y->data[i].v)
                {
                    correct_cnt++;
                }
            }

            return correct_cnt;
        }

        int Model::regression_sigmoid_accuracy_fn(Tensor *p, Tensor *y, int batch_size)
        {
            int correct_cnt = 0;

            for (int i = 0; i < batch_size; i++)
            {
                float p_val = p->data[i].v >= 0.50 ? 1.0f : 0.0f;

                if (p_val == y->data[i].v)
                {
                    correct_cnt++;
                }
            }

            return correct_cnt;
        }

        int Model::regression_tanh_accuracy_fn(Tensor *p, Tensor *y, int batch_size)
        {
            int correct_cnt = 0;

            for (int i = 0; i < batch_size; i++)
            {
                float p_val = p->data[i].v;
                if (p_val < 0.0f)
                {
                    p_val = -1.0f;
                }
                else if (p_val > 0.0f)
                {
                    p_val = 1.0f;
                }

                if (p_val == y->data[i].v)
                {
                    correct_cnt++;
                }
            }

            return correct_cnt;
        }

        int Model::classification_accuracy_fn(Tensor *p, Tensor *y, int batch_size)
        {
            int correct_cnt = 0;

            int output_cnt = p->count() / batch_size;

            for (int i = 0; i < batch_size; i++)
            {
                float max_val = p->data[i * output_cnt + 0].v;
                int max_idx = 0;
                for (int j = 1; j < output_cnt; j++)
                {
                    float val = p->data[i * output_cnt + j].v;
                    if (val > max_val)
                    {
                        max_val = val;
                        max_idx = j;
                    }
                }

                if (y->data[i * output_cnt + max_idx].v == 1.0f)
                {
                    correct_cnt++;
                }
            }

            return correct_cnt;
        }
    }
}