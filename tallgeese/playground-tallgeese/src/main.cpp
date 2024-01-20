#include <iostream>

#include <tallgeese/mod.h>

#include "mnist.h"

using namespace tallgeese::core;
using namespace tallgeese::nn;

int main(int argc, char **argv)
{
	auto model = new Model(LossType::MSE, true);

	int batch_size = 8;

	// model->conv2d({8, 1, 28, 28}, {4, 1, 3, 3});
	// model->activation(ActivationType::Sigmoid);
	// model->conv2d({4, 4, 3, 3});
	// model->activation(ActivationType::Sigmoid);
	// model->flatten();
	// model->linear(32);
	// model->activation(ActivationType::Sigmoid);
	// model->linear(10);
	// model->activation(ActivationType::Sigmoid);

	// auto x = Tensor::random({batch_size, 1, 28, 28});
	// auto y = Tensor::zeros({batch_size, 10});
	// y->data[5].v = 1.0f;

	model->flatten({batch_size, 1, 28, 28});
	model->linear(16);
	model->activation(ActivationType::Sigmoid);
	model->linear(16);
	model->activation(ActivationType::Sigmoid);
	model->linear(10);
	model->activation(ActivationType::Sigmoid);

	mnist::train_mnist(model, batch_size, 3);
	mnist::test_mnist(model);

	// model->test(x, y);

	delete model;

	return 0;
}