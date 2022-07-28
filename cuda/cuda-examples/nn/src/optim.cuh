#pragma once

#include <ndarray.cuh>
#include "layer.cuh"

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

        virtual void step() = 0;
    };

    class SGD : public Optimizer
    {
    public:
        SGD(std::vector<Parameters *> model_params_, float learning_rate);

        virtual void step() override;
    };

    class SGDMomentum : public Optimizer
    {
    private:
        std::vector<NdArray *> vdws_;
        std::vector<NdArray *> vdbs_;

    public:
        SGDMomentum(std::vector<Parameters *> model_params_, float learning_rate);
        ~SGDMomentum();

        virtual void step() override;
    };
}
