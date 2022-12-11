#pragma once

#include "learnable.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        namespace layer
        {
            class Linear : public Learnable
            {
            public:
                Linear(bool shared_params, Shape in_shape, Shape out_shape, ActivationType activation);

                virtual void evaluate(Tensor *out) override;
                virtual void derive(Tensor *in, Tensor *in_n) override;

                virtual Shape input_shape() override;
                virtual Shape output_shape() override;

                virtual Layer *copy() override;

                virtual void validate() override;

                virtual void summarize() override;

                int weight_rows();
                int weight_cols();
            };
        }
    }
}