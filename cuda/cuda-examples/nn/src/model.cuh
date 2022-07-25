#include "layer.cuh"
#include "loss.cuh"

using namespace layer;
using namespace loss;

class Model
{
protected:
    std::vector<Layer *> lyrs_;
    Loss *loss_;

    void add_layer(Layer *lyr);

public:
    Model();
    ~Model();

    void linear(int in_cnt, int out_cnt);
    void sigmoid(int in_cnt);

    NdArray *forward(NdArray *x);
    void backward(NdArray *p, NdArray *y);
    float loss(NdArray *p, NdArray *y);
    void step();
};