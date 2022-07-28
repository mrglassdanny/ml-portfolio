#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

int main(int argc, char **argv)
{
	NdArray *a = new NdArray(false, 2, 3);
	a->rands(0.0f, 1.0f);
	a->print();

	a->change_dim(0, 10);
	a->zeros();
	a->print();


	printf("Hello world!");
	return 0;
}