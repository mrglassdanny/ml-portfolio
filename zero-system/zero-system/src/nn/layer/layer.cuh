#pragma once

#include "../../core/mod.cuh"

namespace nn
{
    namespace layer
    {
        class Layer
        {
        protected:
            NdArray *n_;
            NdArray *dn_;

        public:
            ~Layer();

            virtual void evaluate(NdArray *out) = 0;
            virtual void derive(NdArray *in, NdArray *in_n) = 0;

            virtual Shape input_shape() = 0;
            virtual Shape output_shape() = 0;

            virtual void validate() = 0;

            virtual void summarize();

            int in_features();
            int out_features();

            int batch_size();
            void change_batch_size(int batch_size);

            NdArray *neurons();
            void copy_neurons(NdArray *n);

            NdArray *neuron_gradients();
            void zero_grad();
        };
    }
}