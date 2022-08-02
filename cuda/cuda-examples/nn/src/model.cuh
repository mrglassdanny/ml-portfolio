#pragma once

#include <ndarray.cuh>

#include "layer.cuh"

using namespace layer;

class Model
{
protected:
    std::vector<Layer *> lyrs_;

    void add_layer(Layer *lyr);
    void lock_batch_size(int batch_size);

public:
    Model();
    ~Model();

    void linear(int in_cnt, int out_cnt);
    void sigmoid(int in_cnt);

    Layer *first_layer();
    Layer *last_layer();
    std::vector<Layer *> layers();
    std::vector<Parameters *> parameters();

    virtual NdArray *forward(NdArray *x);
};