#pragma once

#include "../core/mod.cuh"

#include "constants.cuh"

#include "layer/mod.cuh"
#include "loss/mod.cuh"
#include "optim/mod.cuh"

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

        NdArray *forward(NdArray *x);
        float loss(NdArray *p, NdArray *y);
        float accuracy(NdArray *p, NdArray *y);
        void backward(NdArray *p, NdArray *y);
        void step();

        Shape input_shape();
        Shape output_shape();

        void validate_layers();
        void validate_loss();
        void validate_optimizer();
        void validate_input(NdArray *x);
        void validate_output(NdArray *y);
        void validate_gradients(NdArray *x, NdArray *y, bool print_params);

        void summarize();

        void add_layer(Layer *lyr);
        void set_loss(Loss *loss);
        void set_optimizer(Optimizer *optim);

        void linear(int out_feature_cnt, ActivationType activation);
        void linear(Shape y_shape, ActivationType activation);
        void linear(int batch_size, int in_feature_cnt, int out_feature_cnt, ActivationType activation);
        void linear(Shape in_shape, int out_feature_cnt, ActivationType activation);
        void conv2d(Shape filter_shape, ActivationType activation);
        void conv2d(Shape filter_shape, Stride stride, ActivationType activation);
        void conv2d(Shape in_shape, Shape filter_shape, Stride stride, ActivationType activation);
        void enhanced_residual(int out_feature_cnt, ActivationType activation);
        void enhanced_residual(Shape y_shape, ActivationType activation);
        void enhanced_residual(int batch_size, int in_feature_cnt, int out_feature_cnt, ActivationType activation);
        void enhanced_residual(Shape in_shape, int out_feature_cnt, ActivationType activation);

        std::vector<Layer *> layers();
        std::vector<Parameters *> parameters();
        Layer *first_layer();
        Layer *last_layer();
        void reset_layer_shapes();

        int batch_size();
        void change_batch_size(int batch_size);

        void performance_check(NdArray *x, NdArray *y, int epoch_cnt);
    };
}
