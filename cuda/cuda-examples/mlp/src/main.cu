
#include "tensor.cuh"

class Layer
{
private:
	Tensor *neurons;

public:
	Layer();
	~Layer();

	virtual Tensor *evaluate(Tensor *x) = 0;
	virtual Tensor *derive(Tensor *dx) = 0;
};

class LinearLayer : Layer
{
private:
	Tensor *weights;
	Tensor *biases;

public:
	LinearLayer();
	~LinearLayer();
};

class Model
{
public:
	Model();
	~Model();

	virtual Tensor *forward(Tensor *x) = 0;
};

class Loss
{
public:
	Loss();
	~Loss();

	virtual float loss(Tensor *p, Tensor *y) = 0;
	virtual void backward() = 0;
};

class Optimizer
{
public:
	Optimizer();
	~Optimizer();

	virtual void step() = 0;
};

int main(int argc, char **argv)
{

	return 0;
}