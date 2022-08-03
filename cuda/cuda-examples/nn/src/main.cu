#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"

using namespace nn;

int main(int argc, char **argv)
{
	auto model = new Model();

	model->add_layer(new Linear(10, 32));
	model->add_layer(new Sigmoid(32));
	model->add_layer(new Linear(32, 16));
	model->add_layer(new Sigmoid(16));
	model->add_layer(new Linear(16, 1));
	model->add_layer(new Sigmoid(1));

	model->set_loss(new MSE());
	model->set_optimizer(new SGD(model->parameters(), 1.0f));

	NdArray *x = NdArray::rands(true, Shape(1, 10), 0.0f, 1.0f);
	NdArray *y = NdArray::ones(true, Shape(1, 1));

	model->grad_check(x, y, true);

	model->performance_check(x, y, 10);

	NdArray::to_csv("temp/x.csv", x);

	NdArray *x2 = NdArray::from_csv("temp/x.csv");
	x2->print();

	delete model;

	return 0;
}