#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();
	model->add_layer(new Conv2d(1, Shape(8, 8), 4, Shape(2, 2), Shape(0, 0), Shape(1, 1)));

	NdArray *x = NdArray::ones(false, Shape(1, 8, 8));

	model->forward(x)->print();



	return 0;
}