#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	//auto model = new Model();
	/*model->conv2d(Shape(1, 2, 16, 16), Shape(2, 2, 2, 2), Stride(2, 2));
	model->conv2d(Shape(2, 2, 2, 2), Stride(2, 2));
	model->conv2d(Shape(2, 2, 2, 2), Stride(2, 2));
	model->linear(16);
	model->linear(4);
	model->linear(1);
	model->set_loss(new MSE());*/

	auto model = new Model();
	model->add_layer(new Conv2d(Shape(1, 2, 16, 16), Shape(2, 2, 2, 2), Padding(0, 0), Stride(2, 2)));
	model->add_layer(new Conv2d(Shape(1, 2, 8, 8), Shape(2, 2, 2, 2), Padding(0, 0), Stride(2, 2)));
	model->add_layer(new Conv2d(Shape(1, 2, 4, 4), Shape(2, 2, 2, 2), Padding(0, 0), Stride(2, 2)));
	model->add_layer(new Linear(8, 16));
	model->add_layer(new Linear(16, 4));
	model->add_layer(new Linear(4, 1));
	model->set_loss(new MSE());

	NdArray *x = NdArray::ones(true, Shape(1, 2, 16, 16));
	NdArray *y = NdArray::ones(true, model->output_shape());

	model->gradient_check(x, y, true);

	return 0;
}