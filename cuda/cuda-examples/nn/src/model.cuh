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

        struct Validations
        {
            bool layers;
            bool loss;
            bool optimizer;
        } validations_;

    public:
        Model();
        ~Model();

        void add_layer(Layer *lyr);
        std::vector<Layer *> layers();
        std::vector<Parameters *> parameters();
        Layer *first_layer();
        Layer *last_layer();
        void validate_layers();

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

        void set_loss(Loss *loss);
        void validate_loss();

        void set_optimizer(Optimizer *optim);
        void validate_optimizer();

        Shape input_shape();
        Shape output_shape();
        void validate_input(NdArray *x);
        void validate_output(NdArray *y);

        NdArray *forward(NdArray *x);
        float loss(NdArray *p, NdArray *y);
        void backward(NdArray *p, NdArray *y);
        void step();

        int batch_size();
        void summarize();
        void gradient_check(NdArray *x, NdArray *y, bool print_params);
        void performance_check(NdArray *x, NdArray *y, int epoch_cnt);
    };
}
