#pragma once

#include "layer/mod.h"

namespace tallgeese
{
    namespace nn
    {
        using namespace layer;

        enum LossType
        {
            MSE,
            CrossEntropy
        };

        class Model
        {
        private:
            ADContext *ctx;
            std::vector<Layer *> layers;
            LossType loss_type;
            float lr;

            Shape get_output_shape();

        public:
            Model(LossType loss_type);
            Model(LossType loss_type, float learning_rate);
            ~Model();

            int get_batch_size();

            Tensor *forward(Tensor *x);
            Var loss(Tensor *p, Tensor *y);
            float accuracy(Tensor *p, Tensor *y, int (*acc_fn)(Tensor *p, Tensor *y, int batch_size));
            void backward();
            void step();
            void reset();
            void test(Tensor *x, Tensor *y, bool print_grads);

            void linear(int batch_size, int inputs, int outputs, bool bias);
            void linear(int batch_size, int inputs, int outputs);
            void linear(int outputs, bool bias);
            void linear(int outputs);

            void conv2d(Shape input_shape, Shape filter_shape, bool bias);
            void conv2d(Shape input_shape, Shape filter_shape);
            void conv2d(Shape filter_shape, bool bias);
            void conv2d(Shape filter_shape);

            void activation(ActivationType type);

            void softmax();

            void flatten(Shape input_shape);
            void flatten();

            static int regression_accuracy_fn(Tensor *p, Tensor *y, int batch_size);
            static int regression_sigmoid_accuracy_fn(Tensor *p, Tensor *y, int batch_size);
            static int regression_tanh_accuracy_fn(Tensor *p, Tensor *y, int batch_size);
            static int classification_accuracy_fn(Tensor *p, Tensor *y, int batch_size);
        };
    }
}