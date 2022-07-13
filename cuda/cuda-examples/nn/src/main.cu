#include <util.cuh>
#include <tensor.cuh>

int main(int argc, char** argv)
{
	StopWatch sw;
	sw.start();

	Tensor *A = new Tensor(false, Dimensions(3, 3));
	Tensor* C = new Tensor(false, Dimensions(3, 2));
	Tensor* B = new Tensor(false, Dimensions(C->get_dims().get_dim(1), A->get_dims().get_dim(1)));

	A->rands(0.0f, 1.0f);
	B->rands(0.0f, 1.0f);
	C->zeros();

	A->print();
	B->print();
	C->print();

	delete A;
	delete B;
	delete C;

	sw.stop();

	sw.print_elapsed_seconds();

	return 0;
}