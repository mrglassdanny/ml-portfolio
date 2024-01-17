#include <iostream>

#include "tallgeese/core/mod.h"

using namespace tallgeese::core;

int main(int argc, char **argv)
{
	ADContext ctx(true);

	auto a = ctx.parm(2.0f);
	auto b = ctx.parm(2.0f);

	auto c = ctx.multiply(a, b);
	auto d = ctx.multiply(c, a);
	auto e = ctx.power(d, ctx.var(4.0f));

	// auto x1 = ctx.var(Tensor::random({10}));
	// auto w1 = ctx.parm(Tensor::random({10}));
	// auto z1 = ctx.dot(x1, w1);
	// auto a1 = ctx.sigmoid(z1);

	ctx.derive();

	ctx.check_gradients();

	return 0;
}