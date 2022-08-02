#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();

	model->add_layer(new Linear(10, 1));
	model->add_layer(new Sigmoid(1));

	model->set_loss(new MSE());
	model->set_optimizer(new SGD(model->parameters(), 1.0f));

	int batch_size = 5;

	NdArray *x = NdArray::ones(true, Shape(batch_size, 10));
	NdArray *y = NdArray::ones(true, Shape(batch_size, 1));

	for (int i = 0; i < 25; i++)
	{
		NdArray* p = model->forward(x);
		NdArray* l = model->loss(p, y);
		model->backward(p, y);
		model->step();

		l->print();
		if (i == 24)
		{
			p->print();
		}

		delete p;
		delete l;
	}

	return 0;
}