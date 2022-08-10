#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();
	model->conv2d(Shape(1, 2, 16, 16), Shape(4, 2, 2, 2), Stride(2, 2));
	model->conv2d(Shape(4, 4, 2, 2), Stride(2, 2));
	model->conv2d(Shape(4, 4, 2, 2), Stride(1, 1));
	model->linear(16);
	model->linear(1);
	model->set_loss(new MSE());

	NdArray *x = NdArray::rands(true, Shape(1, 2, 16, 16), 0.0f, 1.0f);
	NdArray *y = NdArray::ones(true, model->output_shape());

	model->gradient_check(x, y, true);

	return 0;
}