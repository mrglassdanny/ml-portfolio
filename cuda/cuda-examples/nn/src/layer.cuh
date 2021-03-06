#pragma once

#include <ndarray.cuh>

namespace layer
{
    struct Gradients
    {
        NdArray *dw;
        NdArray *db;
    };

    class Parameters
    {
    private:
        NdArray *w_;
        NdArray *b_;
        Gradients *grads_;

    public:
        Parameters(Shape w_shape, Shape b_shape, int fan_in, int fan_out);
        ~Parameters();

        void zero_grad();

        NdArray *weights();
        NdArray *biases();
        NdArray *weight_gradients();
        NdArray *bias_gradients();
    };

    class Layer
    {
    protected:
        NdArray *n_;
        Shape base_shape_;

    public:
        ~Layer();

        virtual void forward(NdArray *out) = 0;
        virtual NdArray *backward(NdArray *in) = 0;

        NdArray *neurons();
        void copy_neurons(NdArray *n);
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

        virtual void forward(NdArray *out) override;
        virtual NdArray *backward(NdArray *in) override;
    };

    class Activation : public Layer
    {
    public:
        Activation(int in_cnt);

        virtual void forward(NdArray *out) = 0;
        virtual NdArray *backward(NdArray *in) = 0;
    };

    class Sigmoid : public Activation
    {
    public:
        Sigmoid(int in_cnt);

        virtual void forward(NdArray *out) override;
        virtual NdArray *backward(NdArray *in) override;
    };
}
