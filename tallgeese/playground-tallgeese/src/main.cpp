#include <iostream>

#include "tallgeese/core/mod.h"
#include "tallgeese/nn/mod.h"

using namespace tallgeese::core;
using namespace tallgeese::nn::layer;

int main(int argc, char **argv)
{
	ADContext ctx(true);

	auto x = ctx.var(Tensor::random({10, 10}));

	auto l1 = new Linear(&ctx, 10, 10, 10, true);
	auto l1_5 = new Activation(&ctx, 10, 10, ActivationType::Sigmoid);
	auto l2 = new Linear(&ctx, 10, 10, 1, true);
	auto l2_5 = new Activation(&ctx, 10, 1, ActivationType::Sigmoid);

	auto z1 = l1->forward(x);
	auto a1 = l1_5->forward(z1);
	auto z2 = l2->forward(a1);
	auto a2 = l2_5->forward(z2);

	ctx.derive();

	ctx.check_gradients();

	return 0;
}