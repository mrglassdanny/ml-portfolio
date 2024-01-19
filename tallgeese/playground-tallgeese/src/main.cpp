#include <iostream>

#include "tallgeese/core/mod.h"
#include "tallgeese/nn/mod.h"

using namespace tallgeese::core;
using namespace tallgeese::nn;

int main(int argc, char **argv)
{
	ADContext ctx(true);

	// Batch size: 10
	// input -> hidden -> output
	// 10 -> 10 -> 1

	auto x = ctx.var(Tensor::random({10, 10}));

	auto l1 = new FullyConnectedLayer(&ctx, 10, 10, 10, true);
	auto l2 = new FullyConnectedLayer(&ctx, 10, 10, 1, true);

	auto z1 = l1->forward(x);

	auto z2 = l2->forward(z1);

	z2->print();

	ctx.derive();

	ctx.check_gradients();

	return 0;
}