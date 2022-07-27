#pragma once

namespace optim
{
    class Optimizer
    {
    private:
        float lr_;

    public:
        Optimizer(float learning_rate);

        virtual void step() = 0;
    };

    class SGD : public Optimizer
    {
    public:
        SGD(float learning_rate);

        virtual void step();
    };
}
