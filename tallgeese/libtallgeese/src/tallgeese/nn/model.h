#pragma once

#include "layer/layer.h"
#include "layer/linear.h"
#include "layer/conv2d.h"
#include "layer/activation.h"
#include "layer/flatten.h"

namespace tallgeese
{
    namespace nn
    {
        using namespace layer;

        class Model
        {
        private:
            ADContext *ctx;
            std::vector<Layer *> layers;

            Shape get_output_shape();

        public:
            Model();
            Model(bool trace);
            ~Model();

            Tensor *forward(Tensor *x);
            void backward();

            void test(Tensor *x);

            void linear(int batch_size, int inputs, int outputs, bool bias);
            void linear(int batch_size, int inputs, int outputs);
            void linear(int outputs, bool bias);
            void linear(int outputs);

            void conv2d(Shape input_shape, Shape filter_shape, bool bias);
            void conv2d(Shape input_shape, Shape filter_shape);
            void conv2d(Shape filter_shape, bool bias);
            void conv2d(Shape filter_shape);

            void activation(ActivationType type);

            void flatten();
        };
    }
}