#pragma once

#include <ndarray.cuh>

namespace nn
{
    namespace layer
    {
        class Layer
        {
        protected:
            NdArray *n_;

        public:
            ~Layer();

            virtual void evaluate(NdArray *out) = 0;
            virtual NdArray *derive(NdArray *in) = 0;

            virtual Shape input_shape() = 0;
            virtual Shape output_shape() = 0;

            int batch_size();

            NdArray *neurons();
            void copy_neurons(NdArray *n);
        };

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

        class Linear : public Learnable
        {
        public:
            Linear(Shape in_shape, Shape out_shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            int in_features();
            int out_features();
            int weight_rows();
            int weight_cols();
        };

        class Padding
        {
        private:
            int row_cnt_;
            int col_cnt_;

        public:
            Padding();
            Padding(int row_cnt, int col_cnt);

            int rows();
            int cols();
        };

        class Stride
        {
        private:
            int row_cnt_;
            int col_cnt_;

        public:
            Stride();
            Stride(int row_cnt, int col_cnt);
            
            int rows();
            int cols();
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

            int channels();
            int in_feature_rows();
            int in_feature_cols();
            int filters();
            int filter_rows();
            int filter_cols();
            int out_feature_rows();
            int out_feature_cols();
        };

        class Activation : public Layer
        {
        public:
            Activation(Shape shape);

            virtual void evaluate(NdArray *out) = 0;
            virtual NdArray *derive(NdArray *in) = 0;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            int features();
        };

        class Sigmoid : public Activation
        {
        public:
            Sigmoid(Shape shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;
        };
    }
}