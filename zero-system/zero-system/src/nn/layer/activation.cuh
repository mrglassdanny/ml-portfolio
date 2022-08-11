#pragma once

#include "layer.cuh"

namespace nn
{
    namespace layer
    {

        class Activation : public Layer
        {
        public:
            Activation(Shape shape);

            virtual void evaluate(NdArray *out) = 0;
            virtual NdArray *derive(NdArray *in) = 0;

            virtual Shape input_shape() override;
            virtual Shape output_shape() override;

            virtual void validate() override;

            int features();
        };

        class Sigmoid : public Activation
        {
        public:
            Sigmoid(Shape shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;
        };

        class Tanh : public Activation
        {
        public:
            Tanh(Shape shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;
        };

        class ReLU : public Activation
        {
        public:
            ReLU(Shape shape);

            virtual void evaluate(NdArray *out) override;
            virtual NdArray *derive(NdArray *in) override;
        };
    }
}