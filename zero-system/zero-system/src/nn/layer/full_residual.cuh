#pragma once

#include "linear.cuh"

namespace nn
{
    namespace layer
    {
        class FullResidual : public Linear
        {
        private:
            std::vector<Parameters *> residual_params_;

        public:
            FullResidual(Shape in_shape, Shape out_shape, ActivationType activation);
            ~FullResidual();

            void evaluate_residual(NdArray *out, int residual_param_idx);
            void derive_residual(NdArray *in, NdArray *in_n, int residual_param_idx);

            void link(Layer *lyr);

            std::vector<Parameters *> residual_parameters();
        };
    }
}