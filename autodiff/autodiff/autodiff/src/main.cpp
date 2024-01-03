#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <random>


class Tensor
{
private:
	std::vector<int> shape;
	float* data;
	std::string fn;

public:

	Tensor(std::vector<int> shape)
	{
		this->shape = shape;
		this->data = (float*)malloc(sizeof(float) * this->size());
		this->zeros();
		this->fn = "N/A";
	}

	Tensor(std::vector<int> shape, float val)
	{
		this->shape = shape;
		this->data = (float*)malloc(sizeof(float) * this->size());
		this->fill(val);
		this->fn = "N/A";
	}

	Tensor(std::vector<int> shape, float mean, float stddev)
	{
		this->shape = shape;
		this->data = (float*)malloc(sizeof(float) * this->size());
		this->random(mean, stddev);
		this->fn = "N/A";
	}

	~Tensor()
	{
		free(this->data);
	}

	void print(const char *nam)
	{
		printf("VARIABLE: %s\n", nam);

		printf("SHAPE: ");
		for (int i = 0; i < this->count(); i++)
		{
			if (i == this->shape.size() - 1)
				printf("%d\n", this->shape[i]);
			else
				printf("%dx", this->shape[i]);
		}

		printf("DATA: ");
		switch (this->shape.size())
		{
		case 1:
			for (int i = 0; i < this->size(); i++)
			{
				printf("%f\t", this->data[i]);
			}
			break;
		case 2:
			for (int i = 0; i < this->shape[0]; i++)
			{
				for (int j = 0; j < this->shape[1]; j++)
				{
					printf("%f\t", this->data[i]);
				}
				printf("\n");
			}
			break;
		default:
			for (int i = 0; i < this->count(); i++)
			{
				printf("%f\t", this->data[i]);
			}
			break;
		}

		printf("\nFUNCTION: ");
		printf(this->fn.c_str());

		printf("\n\n");
	}

	int count()
	{
		return this->shape.size();
	}

	int size()
	{
		int num = 1;
		for (auto dim : this->shape) 
		{
			num *= dim;
		}
		return num;
	}

	void zeros()
	{
		memset(this->data, 0, sizeof(float) * this->size());
	}

	void fill(float val)
	{
		for (int i = 0; i < this->size(); i++)
		{
			this->data[i] = val;
		}
	}

	void random(float mean, float stddev)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		for (int i = 0; i < this->size(); i++)
		{
			std::normal_distribution<float> d(mean, stddev);
			this->data[i] = d(gen);
		}
	}

	static Tensor* dot(Tensor* a, Tensor* b)
	{
		auto t = new Tensor({ 1 });
		for (int i = 0; i < a->size(); i++)
		{
			t->data[0] += a->data[i] * b->data[i];
		}
		t->fn = "DOT PRODUCT";
		return t;
	}

	static Tensor* sigmoid(Tensor *a)
	{
		auto t = new Tensor(a->shape);
		for (int i = 0; i < a->size(); i++)
		{
			t->data[i] = (1.0f / (1.0f + exp(-(a->data[i]))));
		}
		t->fn = "SIGMOID";
		return t;
	}

	static Tensor* mse(Tensor* p, Tensor* y)
	{
		auto t = new Tensor(p->shape);
		for (int i = 0; i < p->size(); i++)
		{
			t->data[i] = (p->data[i] - y->data[i]) * (p->data[i] - y->data[i]);
		}
		t->fn = "MEAN SQUARED ERROR";
		return t;
	}

	static Tensor* grad(Tensor *d)
	{

	}
};


int main(int argc, char **argv)
{

	auto x = new Tensor({ 5 }, 0, 1.0f);
	auto y = new Tensor({ 1 }, 0, 1.0f);

	auto w = new Tensor({ 5 }, 0, 0.01f);

	x->print("x");
	w->print("w");

	auto z = Tensor::dot(x, w);
	z->print("z");

	auto h = Tensor::sigmoid(z);
	h->print("h");

	y->print("y");

	auto l = Tensor::mse(h, y);
	l->print("l");

	return 0;
}