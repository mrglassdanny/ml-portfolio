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

            void evaluate(NdArray *out, int idx);
            void derive(NdArray *in, NdArray *in_n, int idx);

            void link(Layer *lyr);

            std::vector<Parameters *> residual_parameters();
        };
    }
}