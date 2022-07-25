#pragma once

#include <ndarray.cuh>

namespace loss
{
    class Loss
    {
    public:
        virtual void evaluate(ArrayNd *p, ArrayNd *y, float *d_out_val) = 0;
        virtual ArrayNd *derive(ArrayNd *p, ArrayNd *y) = 0;
    };

    class MeanSquaredError : public Loss
    {
    public:
        virtual void evaluate(ArrayNd *p, ArrayNd *y, float *d_out_val) override;
        virtual ArrayNd *derive(ArrayNd *p, ArrayNd *y) override;
    };
}
