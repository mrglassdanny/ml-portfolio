#include <stdio.h>

#include <nn/mod.cuh>

int main(int argc, char **argv)
{
	printf("MNIST-ZERO\n\n");

	auto arr = NdArray::rands(false, Shape(3, 3, 3), 0.0f, 1.0f);
	//auto arr = NdArray::ones(false, Shape(3, 3, 3));

	arr->print();

	return 0;
}