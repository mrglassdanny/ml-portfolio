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
            void lock_batch_size(int batch_size);

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
            Linear(int in_cnt, int out_cnt);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;
        };

        class Conv2d : public Learnable
        {
        private:
            int channel_cnt_;
            Shape in_shape_;
            int filter_cnt_;
            Shape filter_shape_;
            Shape padding_shape_;
            Shape stride_shape_;

        public:
            Conv2d(int channel_cnt, Shape in_shape, int filter_cnt, Shape filter_shape, Shape padding_shape, Shape stride_shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;
        };

        class Activation : public Layer
        {
        public:
            Activation(int in_cnt);

            virtual void evaluate(NdArray *out) = 0;
            virtual NdArray *derive(NdArray *in) = 0;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;
        };

        class Sigmoid : public Activation
        {
        public:
            Sigmoid(int in_cnt);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;
        };
    }
}