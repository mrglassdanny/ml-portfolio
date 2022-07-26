#include <util.cuh>
#include <ndarray.cuh>

#include "model.cuh"


int main(int argc, char **argv)
{
	Model *model = new Model(new loss::MSE(), new optim::Adam(0.001f));

	model->linear(10, 1);
	model->sigmoid(1);


	delete model;

	return 0;
}