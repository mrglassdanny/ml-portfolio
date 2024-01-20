#include <iostream>

#include <tallgeese/mod.h>

#include "mnist.h"

using namespace tallgeese::core;
using namespace tallgeese::nn;

int main(int argc, char **argv)
{
	auto model = new Model(LossType::MSE);

	int batch_size = 8;

	// model->conv2d({batch_size, 1, 28, 28}, {4, 1, 3, 3});
	// model->activation(ActivationType::Relu);
	// model->conv2d({4, 4, 3, 3});
	// model->activation(ActivationType::Relu);
	// model->flatten();
	// model->linear(16);
	// model->activation(ActivationType::Relu);
	// model->linear(16);
	// model->activation(ActivationType::Relu);
	// model->linear(10);
	// model->activation(ActivationType::Sigmoid);

	model->flatten({batch_size, 1, 28, 28});
	model->linear(16);
	model->activation(ActivationType::Relu);
	model->linear(16);
	model->activation(ActivationType::Relu);
	model->linear(10);
	model->activation(ActivationType::Sigmoid);
	model->softmax();

	mnist::train_mnist(model, batch_size, 5);
	mnist::test_mnist(model);

	// auto x = Tensor::random({batch_size, 1, 28, 28});
	// auto y = Tensor::zeros({batch_size, 10});
	// y->data[5].v = 1.0f;
	// model->test(x, y, false);

	delete model;

	return 0;
}