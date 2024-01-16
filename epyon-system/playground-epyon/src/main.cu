

#include <iostream>

#include <epyon/mod.cuh>

using namespace epyon::core;

int main(int argc, char **argv)
{
	Context ctx(false);

	auto a = ctx.var(1.0f);
	auto b = ctx.var(2.0f);

	auto c = ctx.add(a, b);
	auto d = ctx.mul(a, c);
	auto e = ctx.exp(d, 2.0f);

	auto t = new Tensor(false, Shape({10, 10}));
	t->random(0.0, 1.0);
	t->print();

	ctx.backward();

	return 0;
}