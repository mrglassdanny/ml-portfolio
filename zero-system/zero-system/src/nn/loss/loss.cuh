#pragma once

#include "../../core/mod.cuh"

namespace nn
{
    namespace loss
    {
        class Loss
        {
        public:
            virtual void evaluate(NdArray* p, NdArray* y, NdArray* out) = 0;
            virtual NdArray* derive(NdArray* p, NdArray* y) = 0;

            virtual void summarize();
        };

        class MSE : public Loss
        {
        public:
            virtual void evaluate(NdArray* p, NdArray* y, NdArray* out) override;
            virtual NdArray* derive(NdArray* p, NdArray* y) override;
        };

        class CrossEntropy : public Loss
        {
        public:
            virtual void evaluate(NdArray *p, NdArray *y, NdArray *out) override;
            virtual NdArray *derive(NdArray *p, NdArray *y) override;
        };
    }
}
