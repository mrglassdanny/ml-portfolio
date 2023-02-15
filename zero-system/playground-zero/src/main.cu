#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

using namespace zero::core;
using namespace zero::nn;

// function: y = 2x + 3

int main(int argc, char **argv)
{
    // DATA SETUP:
    auto train_x = Tensor::from_csv("data/train-x.csv");
    auto train_y = Tensor::from_csv("data/train-y.csv");
    {
        train_x->reshape(Shape(train_x->count(), 1));
        train_y->reshape(Shape(train_y->count(), 1));
    }

    auto test_x = Tensor::from_csv("data/test-x.csv");
    auto test_y = Tensor::from_csv("data/test-y.csv");
    {
        test_x->reshape(Shape(test_x->count(), 1));
        test_y->reshape(Shape(test_y->count(), 1));
    }

    // MODEL SETUP:
    float learning_rate = 0.01f;
    auto model = new Model(new MSE(), new Xavier());
    model->linear(train_x->shape(), 1, nullptr);
    model->set_optimizer(new SGD(model->parameters(), learning_rate));

    // TRAIN:
    {
        FILE *train_csv = fopen("temp/train.csv", "w");
        fprintf(train_csv, "epoch,weight,bias,weight_derivative,bias_derivative,loss\n");
        for (int epoch = 1; epoch <= 1000; epoch++)
        {
            auto p = model->forward(train_x);
            auto l = model->loss(p, train_y);
            printf("LOSS: %f\n", l);
            model->backward(p, train_y);

            fprintf(train_csv, "%d,%f,%f,%f,%f,%f\n", epoch, model->parameters()[0]->weights()->get_val(0), model->parameters()[0]->biases()->get_val(0),
                    model->parameters()[0]->weight_gradients()->get_val(0) / model->batch_size() * learning_rate, model->parameters()[0]->bias_gradients()->get_val(0) / model->batch_size() * learning_rate, l);

            model->step();

            delete p;
        }
        fclose(train_csv);
    }

    // TEST:
    {
        model->change_batch_size(test_x->shape()[0]);
        auto p = model->forward(test_x);
        Tensor::to_csv("temp/test_p.csv", p);
        delete p;
    }

    delete train_x;
    delete train_y;

    return 0;
}