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
    protected:
        std::vector<Layer *> lyrs_;
        Loss *loss_;
        Optimizer *optim_;

        Layer *first_layer();
        Layer *last_layer();

    public:
        Model();
        ~Model();

        NdArray *forward(NdArray *x);
        float loss(NdArray *p, NdArray *y);
        void backward(NdArray *p, NdArray *y);
        void step();

        Shape input_shape();
        Shape output_shape();

        void add_layer(Layer *lyr);
        void set_loss(Loss *loss);
        void set_optimizer(Optimizer *optim);

        void linear(int out_feature_cnt);
        void linear(int batch_size, int in_feature_cnt, int out_feature_cnt);
        void conv2d(Shape filter_shape);
        void conv2d(Shape filter_shape, Stride stride);
        void conv2d(Shape filter_shape, Padding padding, Stride stride);
        void conv2d(Shape in_shape, Shape filter_shape, Stride stride);
        void conv2d(Shape in_shape, Shape filter_shape, Padding padding, Stride stride);
        void sigmoid();
        void tanh();
        void relu();

        std::vector<Layer *> layers();
        std::vector<Parameters *> parameters();

        int batch_size();

        void gradient_check(NdArray *x, NdArray *y, bool print_params);
        void performance_check(NdArray *x, NdArray *y, int epoch_cnt);
    };
}
