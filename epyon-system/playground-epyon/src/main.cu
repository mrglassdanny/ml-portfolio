

#include <iostream>

#include <epyon/mod.cuh>

using namespace epyon::core;

int main(int argc, char **argv)
{
	AutoDiffContext ctx(true);

	// auto a = ctx.var(1.0f);
	// auto b = ctx.var(2.0f);

	// auto c = ctx.add(a, b);
	// auto d = ctx.mul(a, c);
	// auto e = ctx.exp(d, 2.0f);

	// ctx.backward();

	auto t = ctx.tensor(Tensor::ones(true, Shape({5})));
	auto s = ctx.tensor(Tensor::zeros(true, Shape({1})));

	ctx.sum(t, s);

	s->print();

	return 0;
}