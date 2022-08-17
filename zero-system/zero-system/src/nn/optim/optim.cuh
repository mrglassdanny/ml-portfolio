#pragma once

#include "../../core/mod.cuh"

#include "../layer/mod.cuh"
#include "../constants.cuh"

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

            virtual void step(int batch_size, int iter) = 0;

            virtual void summarize();
        };

        class SGD : public Optimizer
        {
        public:
            SGD(std::vector<Parameters *> model_params_, float learning_rate);

            virtual void step(int batch_size, int iter) override;
        };

        class SGDMomentum : public Optimizer
        {
        private:
            float beta1_;
            std::vector<NdArray *> vdws_;
            std::vector<NdArray *> vdbs_;

        public:
            SGDMomentum(std::vector<Parameters *> model_params_, float learning_rate, float beta1);
            ~SGDMomentum();

            virtual void step(int batch_size, int iter) override;
        };

        class Adam : public Optimizer
        {
        private:
            float beta1_;
            float beta2_;
            std::vector<NdArray *> vdws_;
            std::vector<NdArray *> vdbs_;
            std::vector<NdArray *> sdws_;
            std::vector<NdArray *> sdbs_;

        public:
            Adam(std::vector<Parameters *> model_params_, float learning_rate, float beta1, float beta2);
            ~Adam();

            virtual void step(int batch_size, int iter) override;
        };
    }
}
