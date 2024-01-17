#include <iostream>

#include "tallgeese/core/mod.h"

using namespace tallgeese::core;

int main(int argc, char **argv)
{
	ADContext ctx(true);

	auto x1 = ctx.var(Tensor::random({10}));
	auto w1 = ctx.parm(Tensor::random({10}));

	auto z1 = ctx.var(Tensor::zeros({1}));
	auto a1 = ctx.var(Tensor::zeros({1}));
	z1 = ctx.dot(x1, w1, z1);
	a1 = ctx.sigmoid(z1, a1);

	auto w2 = ctx.parm(Tensor::random({1}));
	auto z2 = ctx.var(Tensor::zeros({1}));
	auto a2 = ctx.var(Tensor::zeros({1}));
	z2 = ctx.dot(a1, w2, z2);
	a2 = ctx.sigmoid(z2, a2);

	ctx.derive();

	ctx.check_gradients();

	return 0;
}