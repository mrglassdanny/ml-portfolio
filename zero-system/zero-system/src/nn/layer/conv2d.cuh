#pragma once

#include "learnable.cuh"
#include "activation.cuh"

namespace nn
{
    namespace layer
    {
        struct Stride
        {
            int row_cnt;
            int col_cnt;
        };

        class Conv2d : public Learnable
        {
        protected:
            Stride stride_;
            int out_row_cnt_;
            int out_col_cnt_;
            ActivationType activation_;

        public:
            Conv2d(Shape in_shape, Shape filter_shape, Stride stride, ActivationType activation);

            virtual void evaluate(NdArray *out) override;
            virtual void derive(NdArray *in, NdArray *in_n) override;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            virtual void validate() override;

            virtual void summarize() override;

            int channels();
            int in_feature_rows();
            int in_feature_cols();
            int filters();
            int filter_rows();
            int filter_cols();
            Shape filter_shape();
            int stride_rows();
            int stride_cols();
            int out_feature_rows();
            int out_feature_cols();
        };

    }
}