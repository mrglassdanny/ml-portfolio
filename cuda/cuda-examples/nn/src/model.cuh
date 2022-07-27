#include "layer.cuh"
#include "loss.cuh"
#include "optim.cuh"

using namespace layer;
using namespace loss;
using namespace optim;

class Model
{
protected:
    std::vector<NdArray *> neurons_;
    std::vector<Layer *> lyrs_;
    Loss *loss_;
    Optimizer *optim_;

    void add_layer(Layer *lyr);

public:
    Model();
    ~Model();

    void linear(int in_cnt, int out_cnt);
    void sigmoid(int in_cnt);

    virtual NdArray *forward(NdArray *x);
    virtual void backward(NdArray *p, NdArray *y);
    virtual float loss(NdArray *p, NdArray *y);
    virtual void step();
};