#pragma once

#include <ndarray.cuh>

#include "activation.cuh"

namespace layer
{
    class Layer
    {
    public:
        virtual void forward(ArrayNd *out) = 0;
        virtual ArrayNd *backward(ArrayNd *in) = 0;

        virtual ArrayNd *n() = 0;
        virtual void set_n(ArrayNd *n) = 0;
    };

    class Linear : public Layer
    {
    private:
        Array2d *n_;
        Array2d *w_;
        Array1d *b_;
        Array2d *dw_;
        Array1d *db_;

    public:
        Linear(int in_cnt, int out_cnt);
        ~Linear();

        virtual void forward(Array2d *out);
        virtual Array2d *backward(Array2d *in);

        virtual Array2d *n();
        virtual void set_n(Array2d *n);
    };

    class Activation : public Layer
    {
    private:
        Array2d *n_;
        activation::Activation *a_;

    public:
        Activation(activation::Activation *a, int in_cnt);
        ~Activation();

        virtual void forward(Array2d *out);
        virtual Array2d *backward(Array2d *in);

        virtual Array2d *n();
        virtual void set_n(Array2d *n);
    };
}
