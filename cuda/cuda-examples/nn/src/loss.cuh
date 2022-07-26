#pragma once

#include <ndarray.cuh>

namespace loss
{
    class Loss
    {
    public:
        virtual void evaluate(NdArray *p, NdArray *y, float *d_out_val) = 0;
        virtual NdArray *derive(NdArray *p, NdArray *y) = 0;
    };

    class MSE : public Loss
    {
    public:
        virtual void evaluate(NdArray *p, NdArray *y, float *d_out_val) override;
        virtual NdArray *derive(NdArray *p, NdArray *y) override;
    };
}
