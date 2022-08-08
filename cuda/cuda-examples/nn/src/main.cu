#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();
	model->add_layer(new Conv2d(Shape(2, 4, 4), Shape(2, 2, 2, 2), Shape(0, 0), Shape(2, 2)));
	model->set_loss(new MSE());

	NdArray *x = NdArray::rands(true, Shape(1, 2, 4, 4), 0.0f, 0.25f);
	NdArray *y = NdArray::ones(true, model->output_shape());

	x->print();
	Parameters * params = model->parameters()[0];
	params->weights()->print();

	auto p = model->forward(x);

	p->print();

	model->gradient_check(x, y, true);



	return 0;
}