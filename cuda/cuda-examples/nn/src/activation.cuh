#pragma once

#include <ndarray.cuh>

namespace activation
{
    class Activation
    {
    public:
        virtual void evaluate(NdArray *in, NdArray *out) = 0;
        virtual void derive(NdArray *in, NdArray *out) = 0;
    };

    class Sigmoid : public Activation
    {
    public:
        virtual void evaluate(NdArray *in, NdArray *out) override;
        virtual void derive(NdArray *in, NdArray *out) override;
    };
}
