#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();
	model->add_layer(new Conv2d(Shape(2, 16, 16), Shape(2, 2, 2, 2), Shape(0, 0), Shape(2, 2)));
	model->add_layer(new Conv2d(Shape(2, 8, 8), Shape(2, 2, 2, 2), Shape(0, 0), Shape(2, 2)));
	model->add_layer(new Conv2d(Shape(2, 4, 4), Shape(2, 2, 2, 2), Shape(0, 0), Shape(2, 2)));
	model->add_layer(new Linear(8, 16));
	model->add_layer(new Linear(16, 4));
	model->add_layer(new Linear(4, 1));
	model->set_loss(new MSE());

	NdArray *x = NdArray::ones(true, Shape(1, 2, 4, 4));
	NdArray *y = NdArray::full(true, model->output_shape(), 3.0f);

	model->gradient_check(x, y, true);

	return 0;
}