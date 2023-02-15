#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

using namespace zero::core;
using namespace zero::nn;

// function: y = 0.25x + 2.95

int main(int argc, char **argv)
{
    auto x = Tensor::from_csv("data/x.csv");
    auto y = Tensor::from_csv("data/y.csv");

    x->reshape(Shape(x->count(), 1));
    y->reshape(Shape(y->count(), 1));

    x->print();
    y->print();

    auto model = new Model(new loss::MSE(), new Xavier());
    model->linear(x->shape(), 1, nullptr);
    model->set_optimizer(new optim::SGD(model->parameters(), 0.0000001f));

    for (int epoch = 0; epoch < 50; epoch++)
    {
        auto p = model->forward(x);
        auto l = model->loss(p, y);
        printf("LOSS: %f\n", l);
        model->backward(p, y);
        model->step();
    }


    return 0;
}