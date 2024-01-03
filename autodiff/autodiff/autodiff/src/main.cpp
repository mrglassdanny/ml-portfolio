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

public:

	// Example: {1, 2, 3, 4}
	Tensor(std::vector<int> shape)
	{
		this->shape = shape;
		this->data = (float*)malloc(sizeof(float) * this->num_elems());
		this->zeros();
	}

	~Tensor()
	{
		free(this->data);
	}

	void print()
	{
		for (int i = 0; i < this->shape.size(); i++)
		{
			if (i == this->shape.size() - 1)
				printf("%d\n", this->shape[i]);
			else
				printf("%dx", this->shape[i]);
		}

		switch (this->num_dims())
		{
		case 1:
			for (int i = 0; i < this->num_elems(); i++)
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
			for (int i = 0; i < this->num_elems(); i++)
			{
				printf("%f\t", this->data[i]);
			}
			break;
		}
	}

	int num_dims()
	{
		return this->shape.size();
	}

	int num_elems()
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
		memset(this->data, 0, sizeof(float) * this->num_elems());
	}

	void randoms(float mean, float stddev)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		for (int i = 0; i < this->num_elems(); i++)
		{
			std::normal_distribution<float> d(mean, stddev);
			this->data[i] = d(gen);
		}
	}
};


int main(int argc, char **argv)
{
	auto t = new Tensor({5, 5});
	t->print();

	t->randoms(0.0f, 1.0f);
	t->print();
	return 0;
}