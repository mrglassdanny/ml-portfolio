#include <iostream>

#include "tallgeese/core/mod.h"

using namespace tallgeese::core;

int main(int argc, char **argv)
{
	ADContext ctx(true);

	// Batch size: 10
	// input -> hidden -> output
	// 10 -> 10 -> 1

	auto x1 = ctx.var(Tensor::random({10, 10}));
	auto w1 = ctx.parm(Tensor::random({10, 10}));

	auto z1 = ctx.var(Tensor::zeros({10, 10}));
	auto a1 = ctx.var(Tensor::zeros({10, 10}));

	z1 = ctx.matrix_multiply(x1, w1, z1);
	a1 = ctx.sigmoid(z1, a1);

	auto w2 = ctx.parm(Tensor::random({10, 1}));

	auto z2 = ctx.var(Tensor::zeros({10, 1}));
	auto a2 = ctx.var(Tensor::zeros({10, 1}));

	z2 = ctx.matrix_multiply(a1, w2, z2);
	a2 = ctx.sigmoid(z2, a2);

	a2->print();

	ctx.derive();

	ctx.check_gradients();

	return 0;
}