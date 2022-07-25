#include "layer.cuh"
#include "loss.cuh"

using namespace layer;
using namespace loss;

class Model
{
private:
    std::vector<Layer *> lyrs_;
    Loss *loss_;

public:
    Model();
    ~Model();

    ArrayNd *forward(ArrayNd *x);
    void backward(ArrayNd *p, ArrayNd *y);
    float loss(ArrayNd *p, ArrayNd *y);
    void step();
};