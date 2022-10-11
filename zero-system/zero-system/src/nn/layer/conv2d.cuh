#pragma once

#include "learnable.cuh"
#include "activation.cuh"

namespace nn
{
    namespace layer
    {
        struct Padding
        {
            int row_cnt;
            int col_cnt;
        };

        struct Stride
        {
            int row_cnt;
            int col_cnt;
        };

        class Conv2d : public Learnable
        {
        private:
            Padding padding_;
            Stride stride_;
            int out_row_cnt_;
            int out_col_cnt_;

        public:
            Conv2d(Shape in_shape, Shape filter_shape, Padding padding, Stride stride, Activation activation);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in, NdArray *in_n) override;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            virtual void validate() override;

            virtual void reset_shape() override;

            virtual void summarize() override;

            int channels();
            int in_feature_rows();
            int in_feature_cols();
            int filters();
            int filter_rows();
            int filter_cols();
            Shape filter_shape();
            int padding_rows();
            int padding_cols();
            int stride_rows();
            int stride_cols();
            int out_feature_rows();
            int out_feature_cols();

            Shape padded_shape();
        };

    }
}