#pragma once

#include "learnable.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        namespace layer
        {
            class HadamardProduct : public Learnable
            {
            public:
                HadamardProduct(bool shared_params, Shape in_shape, int filter_cnt, Activation *activation, Initializer *initializer);

                virtual void evaluate(Tensor *out) override;
                virtual void derive(Tensor *in, Tensor *in_n) override;

                virtual Shape input_shape() override;
                virtual Shape output_shape() override;

                virtual Layer *copy() override;

                virtual void validate() override;

                virtual void summarize() override;

                int channels();
                int rows();
                int cols();
                int filters();
            };
        }
    }
}