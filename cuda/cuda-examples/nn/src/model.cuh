#include "layer.cuh"

class Model
{
private:
    std::vector<Layer *> layers;

public:
    Model();
    ~Model();

    Tensor *forward(Tensor *x);
};

class Loss
{
private:
    Model &model;

public:
    Loss(Model &model);
    ~Loss();

    virtual float compute(Tensor *p, Tensor *y) = 0;
    virtual void backward(Tensor *p, Tensor *y) = 0;
};

class Optimizer
{
private:
    Model &model;

public:
    Optimizer(Model &model);
    ~Optimizer();

    virtual void step() = 0;
};