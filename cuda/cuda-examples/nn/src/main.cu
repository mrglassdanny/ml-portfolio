#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"
#include "loss.cuh"
#include "optim.cuh"

int main(int argc, char **argv)
{
	auto model = new Model();

	model->linear(10, 20);
	model->sigmoid(20);
	model->linear(20, 4);
	model->sigmoid(4);
	model->linear(4, 1);
	model->sigmoid(1);

	auto loss = new loss::MSE(model->layers());
	auto optim = new optim::SGD(model->parameters(), 1.0f);

	int batch_size = 5;

	NdArray *x = new NdArray(true, batch_size, 10);
	NdArray *y = new NdArray(true, batch_size, 1);

	x->ones();
	y->ones();

	for (int i = 0; i < 25; i++)
	{
		NdArray* p = model->forward(x);
		NdArray* l = loss->loss(p, y);
		loss->backward(p, y);
		optim->step(batch_size);

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