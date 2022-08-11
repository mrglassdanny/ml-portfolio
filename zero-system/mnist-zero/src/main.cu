#include <stdio.h>

#include <core/mod.cuh>

int main(int argc, char **argv)
{
	printf("MNIST-ZERO");
	auto arr = NdArray::rands(false, Shape(10, 10), 0.0f, 1.0f);
	arr->print();
	return 0;
}