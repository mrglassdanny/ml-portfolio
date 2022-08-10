#pragma once

#include <ndarray.cuh>

#include "layer.cuh"

namespace nn
{
    namespace optim
    {
        using namespace layer;

        class Optimizer
        {
        protected:
            std::vector<Parameters *> model_params_;
            float lr_;

        public:
            Optimizer(std::vector<Parameters *> model_params_, float learning_rate);

            virtual void step(int batch_size) = 0;

            virtual void summarize();
        };

        class SGD : public Optimizer
        {
        public:
            SGD(std::vector<Parameters *> model_params_, float learning_rate);

            virtual void step(int batch_size) override;
        };

        class SGDMomentum : public Optimizer
        {
        private:
            float momentum_;
            std::vector<NdArray *> vdws_;
            std::vector<NdArray *> vdbs_;

        public:
            SGDMomentum(std::vector<Parameters *> model_params_, float learning_rate, float momentum);
            ~SGDMomentum();

            virtual void step(int batch_size) override;
        };
    }
}
