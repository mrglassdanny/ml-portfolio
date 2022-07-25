#pragma once

#include <ndarray.cuh>

#include "activation.cuh"

namespace layer
{
    class Layer
    {
    public:
        virtual void forward(NdArray *out) = 0;
        virtual NdArray *backward(NdArray *in) = 0;

        virtual NdArray *n() = 0;
        virtual void set_n(NdArray *n) = 0;
    };

    class Linear : public Layer
    {
    private:
        NdArray *n_;
        NdArray *w_;
        NdArray *b_;
        NdArray *dw_;
        NdArray *db_;

    public:
        Linear(int in_cnt, int out_cnt);
        ~Linear();

        virtual void forward(NdArray *out) override;
        virtual NdArray *backward(NdArray *in) override;

        virtual NdArray *n();
        virtual void set_n(NdArray *n);
    };

    class Activation : public Layer
    {
    private:
        NdArray *n_;
        activation::Activation *a_;

    public:
        Activation(activation::Activation *a, int in_cnt);
        ~Activation();

        virtual void forward(NdArray *out);
        virtual NdArray *backward(NdArray *in);

        virtual NdArray *n();
        virtual void set_n(NdArray *n);
    };
}
