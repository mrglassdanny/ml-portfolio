#pragma once

#include <ndarray.cuh>

namespace layer
{
    class Layer
    {
    protected:
        NdArray *n_;

    public:
        ~Layer();

        virtual void forward(NdArray *out) = 0;
        virtual NdArray * backward(NdArray *in) = 0;

        NdArray *n();
        void set_n(NdArray *n);
    };

    class Learnable : public Layer
    {
    protected:
        NdArray *w_;
        NdArray *b_;
        NdArray *dw_;
        NdArray *db_;

    public:
        ~Learnable();
    };

    class Linear : public Learnable
    {
    public:
        Linear(int in_cnt, int out_cnt);

        virtual void forward(NdArray *out) override;
        virtual NdArray * backward(NdArray *in) override;
    };

    class Sigmoid : public Layer
    {
    public:
        Sigmoid(int in_cnt);

        virtual void forward(NdArray *out) override;
        virtual NdArray * backward(NdArray *in) override;
    };
}
