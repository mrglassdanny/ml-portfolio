#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <random>
#include <functional>


class Var
{
public:
	float v;
	float dv;
	std::function<void(void)> df;
	std::vector<Var*> prev;
	std::string op;

	Var(float v)
	{
	}

	Var(float v, std::vector<Var*> prev)
	{
		this->v = v;
		this->dv = 0.0f;
		this->prev = prev;
		this->op = op;
	}

	Var operator+(const Var& other)
	{
		Var out = Var(this->v + other.v);

		this->df = []() { int i = 0; };
	}
};


class Node
{
public:
	float v;
	float dv;
	std::vector<Node*> parents;

	Node(float v)
	{
		this->v = v;
		this->dv = 0.0f;
	}

	Node(float v, float dv)
	{
		this->v = v;
		this->dv = dv;
	}

	Node(float v, std::vector<Node *> parents)
	{
		this->v = v;
		this->dv = 0.0f;
		this->parents = parents;
	}

	void derive(float dv)
	{
		this->dv += dv;
		for (auto p : this->parents)
		{
			p->derive(dv * this->dv);
		}
	}

	void backprop()
	{
		this->derive(1.0f);
	}

	Node *operator+(const Node &other)
	{
		auto node = new Node(this->v + other.v, { new Node(this->v, 1.0f), new Node(other.v, 1.0f) });
		return node;
	}

	Node *operator*(const Node& other)
	{
		auto node = new Node(this->v * other.v, {new Node(this->v, other.v), new Node(other.v, this->v)});
		return node;
	}

	Node operator-()
	{
		return new Node(-1.0f) * this;
	}

	Node operator-(Node& other)
	{
		return *this * (-other);
	}

	Node operator^(const float pwr)
	{
		auto node = Node(pow(this->v, pwr));
		node.dv = pwr * pow(this->v, pwr - 1);
		return node;
	}

	Node sigmoid()
	{
		auto node = Node((1.0f / (1.0f + exp(-(this->v)))), (this->v * (1.0f - this->v)));
		return node;
	}

	void print()
	{
		printf("(%f, %f)[%d]", this->v, this->dv, this->parents.size());
	}
};

class Tensor
{

public:

	std::vector<int> shape;
	Node* data;

	Tensor(std::vector<int> shape)
	{
		this->shape = shape;
		this->data = (Node*)malloc(sizeof(Node) * this->size());
		this->zeros();
		this->fn = "N/A";
	}

	Tensor(std::vector<int> shape, float val)
	{
		this->shape = shape;
		this->data = (Node*)malloc(sizeof(Node) * this->size());
		this->fill(val);
		this->fn = "N/A";
	}

	Tensor(std::vector<int> shape, float mean, float stddev)
	{
		this->shape = shape;
		this->data = (Node*)malloc(sizeof(Node) * this->size());
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
				this->data[i].print();
				printf("\t");
			}
			break;
		case 2:
			for (int i = 0; i < this->shape[0]; i++)
			{
				for (int j = 0; j < this->shape[1]; j++)
				{
					this->data[i].print();
					printf("\t");
				}
				printf("\n");
			}
			break;
		default:
			for (int i = 0; i < this->count(); i++)
			{
				this->data[i].print();
				printf("\t");
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
		this->fill(0.0f);
	}

	void fill(float val)
	{
		for (int i = 0; i < this->size(); i++)
		{
			this->data[i] = Node(val);
		}
	}

	void random(float mean, float stddev)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		for (int i = 0; i < this->size(); i++)
		{
			std::normal_distribution<float> d(mean, stddev);
			auto n = Node(d(gen));
			this->data[i] = Node(d(gen));
		}
	}

	static Tensor* dot(Tensor* a, Tensor* b)
	{
		auto t = new Tensor({ 1 });
		for (int i = 0; i < a->size(); i++)
		{
			auto c = a->data[i] * b->data[i];
			t->data[0] = t->data[0] + c;
		}
		return t;
	}

	static Tensor* sigmoid(Tensor *a)
	{
		auto t = new Tensor(a->shape);
		for (int i = 0; i < a->size(); i++)
		{
			t->data[i] = a->data[i].sigmoid();
		}
		return t;
	}

	static Tensor* mse(Tensor* p, Tensor* y)
	{
		auto t = new Tensor(p->shape);
		for (int i = 0; i < p->size(); i++)
		{
			t->data[i] = (p->data[i] - y->data[i]) * (p->data[i] - y->data[i]);
		}
		return t;
	}
};


int main(int argc, char **argv)
{
	auto x = new Tensor({ 5 }, 0, 1.0f);
	auto y = new Tensor({ 1 }, 0, 1.0f);

	auto w = new Tensor({ 5 }, 0, 0.01f);

	auto z = Tensor::dot(x, w);
	auto h = Tensor::sigmoid(z);
	//auto l = Tensor::mse(h, y);

	// Gradient Check
	{
		for (int i = 0; i < 5; i++)
		{
			auto ow = w->data[i].v;

			w->data[i].v = ow - 0.001f;
			auto lz = Tensor::dot(x, w);
			auto lh = Tensor::sigmoid(lz);
			//auto ll = Tensor::mse(lh, y);

			w->data[i].v = ow + 0.001f;
			auto rz = Tensor::dot(x, w);
			auto rh = Tensor::sigmoid(rz);
			//auto rl = Tensor::mse(rh, y);

			// float num_grad = (rl->data[0].v - ll->data[0].v) / (2.0f * 0.001f);
			float num_grad = (rh->data[0].v - lh->data[0].v) / (2.0f * 0.001f);
			printf("NUM: %f\n", num_grad);
			printf("ANA: %f\n\n", w->data[i].dv);

			w->data[i].v = ow;
		}
	}

	return 0;
}