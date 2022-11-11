#pragma once

#include "learnable.cuh"
#include "activation.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        namespace layer
        {
            class Linear : public Learnable
            {
            protected:
                ActivationType activation_;

            public:
                Linear();
                Linear(Shape in_shape, Shape out_shape, ActivationType activation);

                virtual void evaluate(Tensor *out) override;
                virtual void derive(Tensor *in, Tensor *in_n) override;

                virtual Shape input_shape() override;
                virtual Shape output_shape() override;

                virtual void validate() override;

                virtual void summarize() override;

                int weight_rows();
                int weight_cols();
            };
        }
    }
}