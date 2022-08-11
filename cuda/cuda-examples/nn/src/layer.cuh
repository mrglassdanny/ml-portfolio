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

            virtual void validate() = 0;
            virtual void summarize();

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

            size_t count();

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

            virtual void validate() override;

            int in_features();
            int out_features();
            int weight_rows();
            int weight_cols();
        };

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

        class Activation : public Layer
        {
        public:
            Activation(Shape shape);

            virtual void evaluate(NdArray *out) = 0;
            virtual NdArray *derive(NdArray *in) = 0;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            virtual void validate() override;

            int features();
        };

        class Sigmoid : public Activation
        {
        public:
            Sigmoid(Shape shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;
        };

        class Tanh : public Activation
        {
        public:
            Tanh(Shape shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;
        };

        class ReLU : public Activation
        {
        public:
            ReLU(Shape shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;
        };
    }
}