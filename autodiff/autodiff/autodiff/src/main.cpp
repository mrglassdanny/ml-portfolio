#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <functional>

enum Oper
{
	None = 0,
	Add,
	Sub,
	Mul,
	Div,
	Pow,
	Sigmoid
};

struct Var
{
	float v = 0.0f;
	float dv = 0.0f;
	Oper oper = None;
	std::vector<Var*> children;
	std::function<void(void)> df = []() { };

	Var() {}

	Var(float v)
	{
		this->v = v;
	}

	void derive()
	{
		this->dv = 1.0f;

		std::set<Var*> s;
		std::vector<Var*> vars;

		std::function<void(Var* var)> trav_f
			= [&](Var *var) {
			if (s.find(var) == s.end())
			{
				s.insert(var);
				for (int i = 0; i < var->children.size(); i++)
				{
					auto child = var->children[i];
					trav_f(child);
				}
				vars.push_back(var);
			}
		};
		trav_f(this);

		for (auto var : vars)
		{
			var->df();
		}
	}

	void print()
	{
		printf("%f (%f): ", this->v, this->dv);
		switch (this->oper)
		{
		case Add:
			printf("+\n");
			break;
		case Mul:
			printf("*\n");
			break;
		case Pow:
			printf("^\n");
			break;
		case Sigmoid:
			printf("o\n");
			break;
		default:
			printf("N/A\n");
			break;
		}
	}
};

Var *add(Var *a, Var *b, Var *out)
{
	if (out == nullptr)
	{
		out = new Var();
	}

	out->v = a->v + b->v;
	out->children.push_back(a);
	out->children.push_back(b);
	out->oper = Add;
	out->df = [a, b, out]() { a->dv += out->dv; b->dv += out->dv; };
	return out;
}

Var *mul(Var* a, Var* b, Var* out)
{
	if (out == nullptr)
	{
		out = new Var();
	}

	out->v = a->v * b->v;
	out->children.push_back(a);
	out->children.push_back(b);
	out->oper = Mul;
	out->df = [a, b, out]() { a->dv += out->dv * b->v; b->dv += out->dv * a->v; };
	return out;
}

Var* mul(Var* a, float num, Var* out)
{
	if (out == nullptr)
	{
		out = new Var();
	}

	auto b = new Var();
	b->v = num;

	out->v = a->v * num;
	out->children.push_back(a);
	out->oper = Mul;
	out->df = [a, num, out]() { a->dv += out->dv * num; };
	return out;
}

Var* pow(Var* a, float num, Var* out)
{
	if (out == nullptr)
	{
		out = new Var();
	}

	out->v = pow(a->v, num);
	out->children.push_back(a);
	out->oper = Pow;
	out->df = [a, num, out]() { a->dv += out->dv * (num * pow(a->v, num - 1)); };
	return out;
}

Var* sig(Var *a, Var * out)
{
	if (out == nullptr)
	{
		out = new Var();
	}

	out->v = (1.0f / (1.0f + exp(-a->v)));
	out->children.push_back(a);
	out->oper = Sigmoid;
	out->df = [a, out]() { a->dv += out->dv * (out->v * (1.0f - out->v)); };
	return out;
}

Var *neg(Var *a, Var *c)
{
	return mul(a, -1.0f, c);
}

Var* sub(Var *a, Var *b, Var *c)
{
	return add(a, neg(b, nullptr), c);
}

Var* div(Var* a, Var* b, Var* c)
{
	return mul(a, pow(b, -1.0f, nullptr), c);
}


class Tensor
{

public:

	std::vector<int> shape;
	Var* data;

	Tensor(std::vector<int> shape)
	{
		this->shape = shape;
		this->data = new Var[this->size()];
		this->zeros();
	}

	Tensor(std::vector<int> shape, float val)
	{
		this->shape = shape;
		this->data = new Var[this->size()];
		this->fill(val);
	}

	Tensor(std::vector<int> shape, float mean, float stddev)
	{
		this->shape = shape;
		this->data = new Var[this->size()];
		this->random(mean, stddev);
	}

	~Tensor()
	{
		delete this->data;
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
			this->data[i] = Var(val);
		}
	}

	void random(float mean, float stddev)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		for (int i = 0; i < this->size(); i++)
		{
			std::normal_distribution<float> d(mean, stddev);
			auto n = Var(d(gen));
			this->data[i] = n;
		}
	}

	static Tensor* dot(Tensor* a, Tensor* b)
	{
		auto t = new Tensor({ 1 });
		for (int i = 0; i < a->size(); i++)
		{
			auto c = mul(&a->data[i], &b->data[i], nullptr);
			add(&t->data[0], c, &t->data[0]);
		}
		return t;
	}

	static Tensor* sigmoid(Tensor *a)
	{
		auto t = new Tensor(a->shape);
		for (int i = 0; i < a->size(); i++)
		{
			sig(&a->data[i], &t->data[i]);
		}
		return t;
	}

	/*static Tensor* mse(Tensor* p, Tensor* y)
	{
		auto t = new Tensor(p->shape);
		for (int i = 0; i < p->size(); i++)
		{
			t->data[i] = (p->data[i] - y->data[i]) * (p->data[i] - y->data[i]);
		}
		return t;
	}*/
};


int main(int argc, char **argv)
{
	auto x = new Tensor({ 5 }, 0, 1.0f);
	auto y = new Tensor({ 1 }, 0, 1.0f);

	auto w = new Tensor({ 5 }, 0, 0.01f);

	auto z = Tensor::dot(x, w);
	auto h = Tensor::sigmoid(z);
	//auto l = Tensor::mse(h, y);

	h->data[0].derive();

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

	/*auto a = Var(2.0f);
	auto b = Var(2.0f);

	auto c = mul(&a, &b, nullptr);
	auto d = pow(c, 2, nullptr);

	d->derive();

	d->print();
	c->print();
	b.print();
	a.print();*/

	return 0;
}