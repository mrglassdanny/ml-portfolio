#include <iostream>

#include "tallgeese/core/mod.h"
#include "tallgeese/nn/mod.h"

using namespace tallgeese::core;
using namespace tallgeese::nn;

int main(int argc, char **argv)
{
	auto model = new Model(LossType::CrossEntropy, true);

	auto x = Tensor::random({2, 1, 8, 8});
	auto y = Tensor::random({2, 4});

	model->conv2d(x->shape, {2, 1, 3, 3});
	model->activation(ActivationType::Tanh);
	model->conv2d({2, 2, 3, 3});
	model->activation(ActivationType::Tanh);
	model->flatten();
	model->linear(4);
	model->activation(ActivationType::Sigmoid);
	model->linear(4);
	model->activation(ActivationType::Sigmoid);
	model->softmax();

	model->test(x, y);

	delete model;

	return 0;
}