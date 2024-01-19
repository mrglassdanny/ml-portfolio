#include "linear.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            Linear::Linear(ADContext *ctx, int batch_size, int inputs, int outputs, bool bias)
                : Layer(ctx)
            {
                this->w = this->ctx->parm(Tensor::random({inputs, outputs}));
                this->b = nullptr;
                this->y = this->ctx->var(Tensor::zeros({batch_size, outputs}));

                if (bias)
                {
                    this->b = this->ctx->parm(Tensor::zeros({outputs}));
                }
            }

            Linear::~Linear()
            {
                delete this->w;
                if (this->b != nullptr)
                {
                    delete this->b;
                }
                delete this->y;
            }

            Tensor *Linear::forward(Tensor *x)
            {
                this->y = this->ctx->matrix_multiply(x, this->w, this->y);
                if (this->b != nullptr)
                {
                    for (int i = 0; i < this->b->count(); i++)
                    {
                        for (int j = 0; j < this->y->shape[0]; j++)
                        {
                            this->y->data[j * this->y->shape[1] + i] = this->ctx->add(
                                this->y->data[j * this->y->shape[1] + i],
                                this->b->data[i]);
                        }
                    }
                }
                return this->y;
            }
        }
    }
}