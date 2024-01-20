#include <iostream>

#include "tallgeese/core/mod.h"
#include "tallgeese/nn/mod.h"

using namespace tallgeese::core;
using namespace tallgeese::nn;

int main(int argc, char **argv)
{
	auto model = new Model(true);

	auto x = Tensor::random({2, 1, 8, 8});

	model->conv2d(x->shape, {2, 1, 3, 3});
	model->activation(ActivationType::Sigmoid);
	model->conv2d({2, 2, 3, 3});
	model->activation(ActivationType::Sigmoid);
	model->flatten();
	model->linear(4);
	model->activation(ActivationType::Sigmoid);
	model->linear(1);
	model->activation(ActivationType::Sigmoid);

	model->test(x);

	delete model;

	return 0;
}