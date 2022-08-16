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
            Shape default_n_shape_;

        public:
            ~Layer();

            virtual void evaluate(NdArray *out) = 0;
            virtual NdArray *derive(NdArray *in) = 0;

            virtual Shape input_shape() = 0;
            virtual Shape output_shape() = 0;

            virtual void validate() = 0;

            virtual void reset_shape();
            
            virtual void summarize();

            int batch_size();
            void change_batch_size(int batch_size);

            NdArray *neurons();
            void copy_neurons(NdArray *n);
        };
    }
}