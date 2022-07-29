#pragma once

#include <ndarray.cuh>

#include "layer.cuh"

namespace loss
{
    using namespace layer;

    class Loss
    {
    protected:
        std::vector<Layer *> lyrs_;

        virtual void evaluate(NdArray *p, NdArray *y, NdArray *out) = 0;
        virtual NdArray *derive(NdArray *p, NdArray *y) = 0;

    public:
        Loss(std::vector<Layer *> layers);

        NdArray *loss(NdArray *p, NdArray *y);
        void backward(NdArray *p, NdArray *y);
    };

    class MSE : public Loss
    {
    protected:
        virtual void evaluate(NdArray *p, NdArray *y, NdArray *out) override;
        virtual NdArray *derive(NdArray *p, NdArray *y) override;

    public:
        MSE(std::vector<Layer *> layers);
    };
}
