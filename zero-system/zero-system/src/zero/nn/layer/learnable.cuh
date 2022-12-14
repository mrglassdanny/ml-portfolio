#pragma once

#include "layer.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        namespace layer
        {
            class Initializer
            {
            public:
                virtual void initialize(Tensor *tensor, int fan_in, int fan_out) = 0;
                virtual Initializer *copy() = 0;

                void summarize();
            };

            class XavierInitializer : public Initializer
            {
            public:
                virtual void initialize(Tensor *tensor, int fan_in, int fan_out) override;
                virtual Initializer *copy() override;
            };

            class HeInitializer : public Initializer
            {
            public:
                virtual void initialize(Tensor *tensor, int fan_in, int fan_out) override;
                virtual Initializer *copy() override;
            };

            class Parameters
            {
            private:
                Tensor *w_;
                Tensor *b_;
                Tensor *dw_;
                Tensor *db_;

            public:
                Parameters(Shape w_shape, Shape b_shape, int fan_in, int fan_out, Initializer *initializer);
                ~Parameters();

                void zero_grad();

                size_t count();

                Tensor *weights();
                Tensor *biases();
                Tensor *weight_gradients();
                Tensor *bias_gradients();
            };

            class Learnable : public Layer
            {
            protected:
                Parameters *params_;
                bool shared_params_;

            public:
                Learnable(bool shared_params);
                ~Learnable();

                Parameters *parameters();
                void share_parameters(Parameters *params);

                void load_parameters();
            };
        }
    }
}