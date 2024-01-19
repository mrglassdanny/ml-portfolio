#include "fully_connected.h"

namespace tallgeese
{
    namespace nn
    {
        FullyConnectedLayer::FullyConnectedLayer(ADContext *ctx, int batch_size, int inputs, int outputs, bool bias)
            : Layer(ctx)
        {
            this->w = this->ctx->parm(Tensor::random({inputs, outputs}));
            this->b = nullptr;
            this->z = this->ctx->var(Tensor::zeros({batch_size, outputs}));

            if (bias)
            {
                this->b = this->ctx->parm(Tensor::zeros({outputs}));
            }
        }

        FullyConnectedLayer::~FullyConnectedLayer()
        {
            delete this->w;
            if (this->b != nullptr)
            {
                delete this->b;
            }
            delete this->z;
        }

        Tensor *FullyConnectedLayer::forward(Tensor *x)
        {
            this->z = this->ctx->matrix_multiply(x, this->w, this->z);
            if (this->b != nullptr)
            {
                for (int i = 0; i < this->b->count(); i++)
                {
                    for (int j = 0; j < this->z->shape[0]; j++)
                    {
                        this->z->data[j * this->z->shape[1] + i] = this->ctx->add(
                            this->z->data[j * this->z->shape[1] + i],
                            this->b->data[i]);
                    }
                }
            }
            return this->z;
        }
    }
}