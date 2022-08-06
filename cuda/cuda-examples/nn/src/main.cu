#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();
	model->add_layer(new Conv2d(1, Shape(4, 4), 4, Shape(2, 2), Shape(0, 0), Shape(1, 1)));

	NdArray *x = NdArray::rands(false, Shape(1, 1, 4, 4), 0.0f, 1.0f);

	model->forward(x)->print();

	return 0;
}