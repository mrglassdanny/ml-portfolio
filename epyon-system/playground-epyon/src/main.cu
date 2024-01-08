

#include <iostream>

#include <epyon/mod.cuh>

using namespace epyon::core;

int main(int argc, char **argv)
{
	Var v;
	v.v = 10;
	Var x;
	x.v = 12;

	Var y;
	y.prev = (Var **)malloc(sizeof(Var *) * 2);
	add(&v, &x, &y);

	int a = 0;
	int b = 0;
	int c = a + b;

	printf("Hello world!");
	return 0;
}