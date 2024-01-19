#pragma once

#include "layer.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            class Conv2d : public Layer
            {
            private:
                Tensor *w;
                Tensor *b;
                Tensor *y;

                Var get_x_var(Tensor *x, int b, int ch, int r, int c);
                Var get_w_var(int f, int ch, int r, int c);
                Var get_y_var(int b, int ch, int r, int c);
                void set_y_var(Var var, int b, int ch, int r, int c);

            public:
                Conv2d(ADContext *ctx, Shape input_shape, Shape filter_shape, bool bias);
                ~Conv2d();

                virtual Tensor *forward(Tensor *x);
                virtual Shape get_output_shape();
            };
        }
    }
}