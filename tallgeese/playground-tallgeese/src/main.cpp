#include <iostream>

#include "tallgeese/core/mod.h"

using namespace tallgeese::core;

int main(int argc, char **argv)
{
	ADContext ctx;

	auto a = ctx.var(2.0f);
	auto b = ctx.var(2.0f);

	auto c = ctx.mul(a, b);
	auto d = ctx.mul(c, c);
	auto e = ctx.pwr(d, 4.0f);

	e.print();

	ctx.derive();
	printf("\nde/da: %f\n", ctx.get_derivative(a));

	return 0;
}