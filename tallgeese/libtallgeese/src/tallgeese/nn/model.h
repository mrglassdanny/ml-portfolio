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

            Shape get_output_shape();

        public:
            Model(LossType loss_type);
            Model(LossType loss_type, bool trace);
            ~Model();

            Tensor *forward(Tensor *x);
            Var loss(Tensor *p, Tensor *y);
            void backward();

            void test(Tensor *x, Tensor *y);

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

            void flatten();
        };
    }
}