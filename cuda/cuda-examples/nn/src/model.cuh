#include "layer.cuh"

class Model
{
private:
    std::vector<Layer *> layers;

public:
    Model();
    ~Model();

    ArrayNd *forward(ArrayNd *x);
};

class Loss
{
private:
    Model &model;

public:
    Loss(Model &model);
    ~Loss();

    virtual float compute(ArrayNd *p, ArrayNd *y) = 0;
    virtual void backward(ArrayNd *p, ArrayNd *y) = 0;
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