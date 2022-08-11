#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();

	 model->conv2d(Shape(1, 2, 16, 16), Shape(4, 2, 2, 2), Stride{2, 2});
	 model->tanh();
	 model->conv2d(Shape(4, 4, 2, 2), Stride{2, 2});
	 model->tanh();
	 model->conv2d(Shape(4, 4, 2, 2), Stride{1, 1});
	 model->tanh();
	 model->linear(16);
	 model->tanh();
	 model->linear(1);
	 model->sigmoid();

	model->set_loss(new MSE());
	model->set_optimizer(new SGD(model->parameters(), 0.001f));

	NdArray *x = NdArray::rands(true, model->input_shape(), 0.0f, 1.0f);
	NdArray *y = NdArray::ones(true, model->output_shape());

	model->summarize();

	return 0;
}