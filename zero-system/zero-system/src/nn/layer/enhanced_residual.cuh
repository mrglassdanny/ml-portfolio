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

            void evaluate(NdArray *out, int idx);
            void derive(NdArray *in, NdArray *in_n, int idx);

            void link(Layer *lyr);

            std::vector<Parameters *> residual_parameters();
        };
    }
}