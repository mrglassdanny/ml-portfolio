#pragma once

#include "layer.cuh"

namespace nn
{
    namespace layer
    {
        class Parameters
        {
        private:
            NdArray *w_;
            NdArray *b_;
            NdArray *dw_;
            NdArray *db_;

        public:
            Parameters(Shape w_shape, Shape b_shape, int fan_in, int fan_out);
            ~Parameters();

            size_t count();

            void zero_grad();

            NdArray *weights();
            NdArray *biases();
            NdArray *weight_gradients();
            NdArray *bias_gradients();
        };

        class Learnable : public Layer
        {
        protected:
            Parameters *params_;

        public:
            ~Learnable();

            Parameters *parameters();
        };
    }
}