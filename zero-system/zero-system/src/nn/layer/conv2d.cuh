#pragma once

#include "learnable.cuh"

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

        public:
            Conv2d(Shape in_shape, Shape filter_shape, Padding padding, Stride stride);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            virtual void validate() override;

            int channels();
            int in_feature_rows();
            int in_feature_cols();
            int filters();
            int filter_rows();
            int filter_cols();
            int padding_rows();
            int padding_cols();
            int stride_rows();
            int stride_cols();
            int out_feature_rows();
            int out_feature_cols();
        };

    }
}