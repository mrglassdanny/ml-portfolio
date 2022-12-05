#pragma once

#include "../core/mod.cuh"

#include "constants.cuh"

#include "layer/mod.cuh"
#include "loss/mod.cuh"
#include "optim/mod.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        using namespace layer;
        using namespace loss;
        using namespace optim;

        class Model
        {
        protected:
            std::vector<Layer *> lyrs_;
            bool shared_params_;
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
            Model(bool shared_params);
            ~Model();

            Tensor *forward(Tensor *x);
            float loss(Tensor *p, Tensor *y);
            float accuracy(Tensor *p, Tensor *y);
            void backward(Tensor *p, Tensor *y);
            void step();

            Shape input_shape();
            Shape output_shape();

            void validate_layers();
            void validate_loss();
            void validate_optimizer();
            void validate_input(Tensor *x);
            void validate_output(Tensor *y);
            void validate_gradients(Tensor *x, Tensor *y, bool print_params);

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
            void hadamard_product(int filter_cnt, ActivationType activation);
            void hadamard_product(Shape in_shape, int filter_cnt, ActivationType activation);
            void matrix_product(int filter_cnt, ActivationType activation);
            void matrix_product(Shape in_shape, int filter_cnt, ActivationType activation);

            std::vector<Layer *> layers();
            std::vector<Parameters *> parameters();
            void share_parameters(std::vector<Parameters *> params);
            Layer *first_layer();
            Layer *last_layer();
            void reset_layer_shapes();

            int batch_size();
            void change_batch_size(int batch_size);

            Optimizer *optimizer();

            void performance_check(Tensor *x, Tensor *y, int epoch_cnt);
        };
    }
}