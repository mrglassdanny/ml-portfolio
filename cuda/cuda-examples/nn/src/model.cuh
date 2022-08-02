#pragma once

#include <ndarray.cuh>

#include "layer.cuh"
#include "loss.cuh"
#include "optim.cuh"

namespace nn
{
    using namespace nn::layer;
    using namespace nn::loss;
    using namespace nn::optim;

    class Model
    {
    protected:
        std::vector<Layer *> lyrs_;
        Loss *loss_;
        Optimizer *optim_;

        Layer *first_layer();
        Layer *last_layer();

        void lock_batch_size(int batch_size);
        int batch_size();

    public:
        Model();
        ~Model();

        void add_layer(Layer *lyr);
        void set_loss(Loss *loss);
        void set_optimizer(Optimizer *optim);

        std::vector<Layer *> layers();
        std::vector<Parameters *> parameters();

        virtual NdArray *forward(NdArray *x);
        virtual NdArray *loss(NdArray *p, NdArray *y);
        virtual void backward(NdArray *p, NdArray *y);
        virtual void step();
    };
}
