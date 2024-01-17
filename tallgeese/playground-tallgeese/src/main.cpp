#include <iostream>

#include "tallgeese/core/mod.h"

using namespace tallgeese::core;

int main(int argc, char **argv)
{
	ADContext ctx(true);

	auto a = ctx.var(2.0f);
	auto b = ctx.var(2.0f);

	auto c = ctx.mul(a, b);
	auto d = ctx.mul(c, a);
	auto e = ctx.pwr(d, ctx.var(4.0f));

	ctx.derive();

	ctx.check_grad();

	return 0;
}