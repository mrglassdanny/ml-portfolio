#pragma once

#include <conio.h>

#include <tallgeese/mod.h>

using namespace tallgeese::core;
using namespace tallgeese::nn;

namespace mnist
{
    struct Batch
    {
        Tensor *x;
        Tensor *y;
    };

    std::vector<Batch> get_train_dataset(int batch_size);
    std::vector<Batch> get_test_dataset(int batch_size);

    void train_mnist(Model *model, int batch_size, int epochs, bool test_each_epoch);
    float test_mnist(Model *model, int epoch, bool train, bool to_file);
}