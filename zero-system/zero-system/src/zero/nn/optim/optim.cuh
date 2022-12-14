#pragma once

#include "../../core/mod.cuh"

#include "../layer/mod.cuh"
#include "../constants.cuh"

namespace zero
{
    using namespace core;

    namespace nn
    {
        using namespace layer;

        namespace optim
        {
            class Optimizer
            {
            protected:
                std::vector<Parameters *> model_params_;
                float lr_;
                int step_num_ = 1;

            public:
                Optimizer(std::vector<Parameters *> model_params, float learning_rate);

                virtual void step(int batch_size) = 0;

                virtual Optimizer *copy() = 0;

                virtual void summarize();

                void set_learning_rate(float learning_rate);
                void scale_learning_rate(float factor);
            };

            class SGD : public Optimizer
            {
            public:
                SGD(std::vector<Parameters *> model_params, float learning_rate);

                virtual void step(int batch_size) override;

                virtual Optimizer *copy() override;
            };

            class SGDMomentum : public Optimizer
            {
            private:
                float beta1_;
                std::vector<Tensor *> mdws_;
                std::vector<Tensor *> mdbs_;

            public:
                SGDMomentum(std::vector<Parameters *> model_params, float learning_rate, float beta1);
                ~SGDMomentum();

                virtual void step(int batch_size) override;

                virtual Optimizer *copy() override;
            };

            class Adam : public Optimizer
            {
            private:
                float beta1_;
                float beta2_;
                std::vector<Tensor *> mdws_;
                std::vector<Tensor *> mdbs_;
                std::vector<Tensor *> vdws_;
                std::vector<Tensor *> vdbs_;

            public:
                Adam(std::vector<Parameters *> model_params, float learning_rate, float beta1, float beta2);
                ~Adam();

                virtual void step(int batch_size) override;

                virtual Optimizer *copy() override;
            };
        }
    }
}