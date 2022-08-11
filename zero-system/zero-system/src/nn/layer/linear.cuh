#pragma once

#include "learnable.cuh"

namespace nn
{
    namespace layer
    {

        class Linear : public Learnable
        {
        public:
            Linear(Shape in_shape, Shape out_shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            virtual void validate() override;

            int in_features();
            int out_features();
            int weight_rows();
            int weight_cols();
        };

    }
}