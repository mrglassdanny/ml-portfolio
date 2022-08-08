#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();
	model->add_layer(new Conv2d(Shape(2, 4, 4), Shape(2, 2, 2, 2), Shape(0, 0), Shape(2, 2)));

	NdArray *x = NdArray::ones(false, Shape(1, 2, 4, 4));
	for (int i = 0; i < x->count(); i++)
	{
		x->set_val(i, rand() % 4);
	}

	x->print();
	Parameters *p = model->parameters()[0];
	p->weights()->print();

	model->forward(x)->print();

	return 0;
}