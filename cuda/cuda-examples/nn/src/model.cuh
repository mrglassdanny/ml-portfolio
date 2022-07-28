#include "layer.cuh"
#include "loss.cuh"
#include "optim.cuh"

using namespace layer;
using namespace loss;
using namespace optim;

class Model
{
protected:
    std::vector<Layer *> lyrs_;
    Loss *loss_;

    void add_layer(Layer *lyr);

public:
    Model(Loss *loss);
    ~Model();

    void linear(int in_cnt, int out_cnt);
    void sigmoid(int in_cnt);

    void lock_batch_size(int batch_size);

    virtual NdArray *forward(NdArray *x);
    virtual void backward(NdArray *p, NdArray *y);
    virtual float loss(NdArray *p, NdArray *y);

    std::vector<Parameters *> parameters();
};