#pragma once

#include "linear.cuh"

namespace nn
{
    namespace layer
    {
        class EnhancedResidual : public Linear
        {
        private:
            std::vector<Parameters *> residual_params_;

        public:
            EnhancedResidual(Shape in_shape, Shape out_shape, ActivationType activation);
            ~EnhancedResidual();

            void evaluate_residual(NdArray *out, int idx);
            void derive_residual(NdArray *in, NdArray *in_n, int idx);

            void compile(std::vector<Layer *> layers, int my_idx);

            std::vector<Parameters *> residual_parameters();
        };
    }
}