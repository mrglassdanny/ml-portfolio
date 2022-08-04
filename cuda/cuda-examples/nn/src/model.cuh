#pragma once

#include <ndarray.cuh>
#include <util.cuh>

#include "layer.cuh"
#include "loss.cuh"
#include "optim.cuh"
#include "constants.cuh"

namespace nn
{
    using namespace nn::layer;
    using namespace nn::loss;
    using namespace nn::optim;

    class Model
    {
    private:
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

        NdArray *forward(NdArray *x);
        float loss(NdArray *p, NdArray *y);
        void backward(NdArray *p, NdArray *y);
        void step();

        void gradient_check(NdArray *x, NdArray *y, bool print_params);
        void performance_check(NdArray *x, NdArray *y, int epoch_cnt);
    };
}
