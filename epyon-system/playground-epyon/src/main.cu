

#include <iostream>

#include <epyon/mod.cuh>

using namespace epyon::core;

int main(int argc, char **argv)
{
	AutoDiffContext ctx(false);

	auto a = ctx.var(1.0f);
	auto b = ctx.var(2.0f);

	auto c = ctx.add(a, b);
	auto d = ctx.mul(a, c);
	auto e = ctx.exp(d, 2.0f);

	auto t = ctx.tensor(Tensor::random(false, Shape({5, 5}), 0.0f, 1.0f));
	t->print();

	ctx.backward();

	return 0;
}