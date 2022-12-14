#pragma once

#include "../../core/mod.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        namespace loss
        {
            class Loss
            {
            public:
                virtual void evaluate(Tensor *p, Tensor *y, Tensor *out) = 0;
                virtual Tensor *derive(Tensor *p, Tensor *y) = 0;

                virtual Loss *copy() = 0;

                virtual void summarize();
            };

            class MSE : public Loss
            {
            public:
                virtual void evaluate(Tensor *p, Tensor *y, Tensor *out) override;
                virtual Tensor *derive(Tensor *p, Tensor *y) override;

                virtual Loss *copy() override;
            };

            class CrossEntropy : public Loss
            {
            public:
                virtual void evaluate(Tensor *p, Tensor *y, Tensor *out) override;
                virtual Tensor *derive(Tensor *p, Tensor *y) override;

                virtual Loss *copy() override;
            };
        }
    }
}