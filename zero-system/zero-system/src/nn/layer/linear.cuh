#pragma once

#include "learnable.cuh"
#include "activation.cuh"

namespace nn
{
    namespace layer
    {

        class Linear : public Learnable
        {
        private:
            Activation activation_;
            
        public:
            Linear(Shape in_shape, Shape out_shape, Activation activation);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in, NdArray *in_n) override;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            virtual void validate() override;

            virtual void summarize() override;

            int in_features();
            int out_features();
            int weight_rows();
            int weight_cols();
        };

    }
}