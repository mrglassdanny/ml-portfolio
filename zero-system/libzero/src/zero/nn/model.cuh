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
            Initializer *initializer_;

            struct Validations
            {
                bool layers;
                bool loss;
                bool optimizer;
            } validations_;

        public:
            Model();
            Model(bool shared_params);
            Model(Initializer *initializer);
            Model(Loss *loss, Initializer *initializer);
            ~Model();

            Model *copy();

            Tensor *forward(Tensor *x);
            float loss(Tensor *p, Tensor *y);
            float accuracy(Tensor *p, Tensor *y, int (*acc_fn)(Tensor *p, Tensor *y, int batch_size));
            void backward(Tensor *p, Tensor *y);
            void step();
            void zero_grad();

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
            void set_initializer(Initializer *initializer);

            void linear(int out_feature_cnt, Activation *activation);
            void linear(Shape y_shape, Activation *activation);
            void linear(int batch_size, int in_feature_cnt, int out_feature_cnt, Activation *activation);
            void linear(Shape in_shape, int out_feature_cnt, Activation *activation);
            void conv2d(Shape filter_shape, Activation *activation);
            void conv2d(Shape filter_shape, Stride stride, Activation *activation);
            void conv2d(Shape in_shape, Shape filter_shape, Stride stride, Activation *activation);

            std::vector<Layer *> layers();

            std::vector<Parameters *> parameters();
            void share_parameters(std::vector<Parameters *> params);
            void save_parameters(const char *path);
            void load_parameters(const char *path);

            Layer *first_layer();
            Layer *last_layer();

            int batch_size();
            void change_batch_size(int batch_size);

            Optimizer *optimizer();

            void performance_check(Tensor *x, Tensor *y, int epoch_cnt);

            static int regression_accuracy_fn(Tensor *p, Tensor *y, int batch_size);
            static int regression_sigmoid_accuracy_fn(Tensor *p, Tensor *y, int batch_size);
            static int regression_tanh_accuracy_fn(Tensor *p, Tensor *y, int batch_size);
            static int classification_accuracy_fn(Tensor *p, Tensor *y, int batch_size);
        };
    }
}