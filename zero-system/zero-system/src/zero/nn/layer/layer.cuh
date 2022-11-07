#pragma once

#include "../../core/mod.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        namespace layer
        {
            class Layer
            {
            protected:
                Tensor *n_;
                Tensor *dn_;

            public:
                ~Layer();

                virtual void evaluate(Tensor *out) = 0;
                virtual void derive(Tensor *in, Tensor *in_n) = 0;

                virtual Shape input_shape() = 0;
                virtual Shape output_shape() = 0;

                virtual void validate() = 0;

                virtual void summarize();

                int in_features();
                int out_features();

                int batch_size();
                void change_batch_size(int batch_size);

                Tensor *neurons();
                void copy_neurons(Tensor *n);

                Tensor *neuron_gradients();
                void zero_grad();
            };
        }
    }
}