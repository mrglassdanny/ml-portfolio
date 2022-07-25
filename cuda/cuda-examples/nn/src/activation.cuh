#pragma once

#include <ndarray.cuh>

namespace activation
{
    class Activation
    {
    public:
        virtual void evaluate(Array2d *in, Array2d *out) = 0;
        virtual void derive(Array2d *in, Array2d *out) = 0;
    };

    class Sigmoid : public Activation
    {
    public:
        virtual void evaluate(Array2d *in, Array2d *out) override;
        virtual void derive(Array2d *in, Array2d *out) override;
    };
}
