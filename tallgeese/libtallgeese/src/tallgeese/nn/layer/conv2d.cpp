#include "conv2d.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            Conv2d::Conv2d(ADContext *ctx, Shape input_shape, Shape filter_shape, bool bias)
                : Layer(ctx)
            {
                this->w = Tensor::random(filter_shape);
                this->b = nullptr;

                int batch_size = input_shape[0];
                int x_channels = input_shape[1];
                int x_rows = input_shape[2];
                int x_cols = input_shape[3];

                int filters = filter_shape[0];
                int filter_channels = filter_shape[1];
                int filter_rows = filter_shape[2];
                int filter_cols = filter_shape[3];

                this->y = Tensor::zeros({batch_size,
                                         filters,
                                         x_rows - filter_rows + 1,
                                         x_cols - filter_cols + 1});

                if (bias)
                {
                    this->b = Tensor::zeros({filters, filter_channels});
                }
            }

            Conv2d::~Conv2d()
            {
                delete this->w;
                if (this->b != nullptr)
                {
                    delete this->b;
                }
                delete this->y;
            }

            Tensor *Conv2d::forward(Tensor *x)
            {
                this->reset();

                this->w = this->ctx->parm(this->w);
                this->y = this->ctx->var(this->y);
                if (this->b != nullptr)
                {
                    this->b = this->ctx->parm(this->b);
                }

                int batch_size = x->shape[0];
                int x_channels = x->shape[1];
                int x_rows = x->shape[2];
                int x_cols = x->shape[3];

                int filters = this->w->shape[0];
                int filter_channels = this->w->shape[1];
                int filter_rows = this->w->shape[2];
                int filter_cols = this->w->shape[3];

                int y_channels = this->y->shape[1];
                int y_rows = this->y->shape[2];
                int y_cols = this->y->shape[3];

                for (int b = 0; b < batch_size; b++)
                {
                    for (int y_ch = 0; y_ch < y_channels; y_ch++)
                    {
                        for (int y_r = 0; y_r < y_rows; y_r++)
                        {
                            for (int y_c = 0; y_c < y_cols; y_c++)
                            {
                                for (int x_ch = 0; x_ch < x_channels; x_ch++)
                                {
                                    for (int f_r = 0; f_r < filter_rows; f_r++)
                                    {
                                        for (int f_c = 0; f_c < filter_cols; f_c++)
                                        {
                                            this->y->set_var(this->ctx->add(
                                                                 this->ctx->multiply(this->w->get_var(y_ch, x_ch, f_r, f_c), x->get_var(b, x_ch, y_r + f_r, y_c + f_c)),
                                                                 this->y->get_var(b, y_ch, y_r, y_c)),
                                                             b, y_ch, y_r, y_c);
                                        }
                                    }

                                    if (this->b != nullptr)
                                    {
                                        this->y->set_var(this->ctx->add(this->y->get_var(b, y_ch, y_r, y_c), this->b->get_var(y_ch, x_ch)),
                                                         b, y_ch, y_r, y_c);
                                    }
                                }
                            }
                        }
                    }
                }

                return this->y;
            }

            Shape Conv2d::get_output_shape()
            {
                return this->y->shape;
            }
        }
    }
}