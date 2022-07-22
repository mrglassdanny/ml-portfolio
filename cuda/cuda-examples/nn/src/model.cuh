#include "layer.cuh"

class Model
{
private:
    std::vector<Layer *> lyrs_;
    Loss *loss_;

public:
    Model();
    ~Model();

    ArrayNd *forward(ArrayNd *x);
    void backward();
    float loss(ArrayNd *p, ArrayNd *y);
    void step();
};

class Loss
{

public:
    virtual float evaluate(ArrayNd *p, ArrayNd *y) = 0;
    virtual void derive(ArrayNd *p, ArrayNd *y) = 0;
};

class MSELoss : public Loss
{
    public:
    virtual float evaluate(ArrayNd *p, ArrayNd *y);
    virtual ArrayNd *derive(ArrayNd *p, ArrayNd *y);
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