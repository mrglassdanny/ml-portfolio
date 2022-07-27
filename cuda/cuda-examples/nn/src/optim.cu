#include "optim.cuh"

using namespace optim;

Optimizer::Optimizer(float learning_rate)
{
    this->lr_ = learning_rate;
}

SGD::SGD(float learning_rate)
    : Optimizer(learning_rate)
{
}

void SGD::step() {}