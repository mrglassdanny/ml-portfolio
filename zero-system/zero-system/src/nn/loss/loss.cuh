#pragma once

#include "../../core/mod.cuh"

namespace nn
{
    namespace loss
    {
        class Loss
        {
        public:
            virtual void evaluate(Tensor *p, Tensor *y, Tensor *out) = 0;
            virtual Tensor *derive(Tensor *p, Tensor *y) = 0;

            virtual void summarize();
        };

        class MSE : public Loss
        {
        public:
            virtual void evaluate(Tensor *p, Tensor *y, Tensor *out) override;
            virtual Tensor *derive(Tensor *p, Tensor *y) override;
        };

        class CrossEntropy : public Loss
        {
        public:
            virtual void evaluate(Tensor *p, Tensor *y, Tensor *out) override;
            virtual Tensor *derive(Tensor *p, Tensor *y) override;
        };
    }
}
