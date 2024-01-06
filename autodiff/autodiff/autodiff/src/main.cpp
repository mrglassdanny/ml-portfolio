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

#define EPSILON 0.001f

enum Oper
{
	None = 0,
	Add,
	Inc,
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

		std::function<void(Var* var)> trav_f = [&](Var *var) 
		{
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

		std::reverse(vars.begin(), vars.end());
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
		case Inc:
			printf("+=\n");
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
	if (a != out)
		out->children.push_back(a);
	if (b != out)
		out->children.push_back(b);
	out->oper = Add;
	out->df = [a, b, out]() 
	{ 
		if (a != out)
			a->dv += out->dv;
		if (b != out)
			b->dv += out->dv; 
	};
	return out;
}

Var* inc(Var* a, Var* out)
{
	if (out == nullptr)
	{
		out = new Var();
	}

	out->v += a->v;
	out->children.push_back(a);
	out->oper = Inc;
	out->df = [a, out]()
		{
			for (auto child : out->children)
			{
				child->dv += out->dv;
			}
		};

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
	out->df = [a, b, out]() 
	{ 
		a->dv += out->dv * b->v; 
		b->dv += out->dv * a->v; 
	};
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
	out->df = [a, num, out]() 
	{ 
		a->dv += out->dv * num; 
	};
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
	out->df = [a, num, out]() 
	{ 
		a->dv += out->dv * (num * pow(a->v, num - 1));
	};
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
	out->df = [a, out]() 
	{ 
		a->dv += out->dv * (out->v * (1.0f - out->v)); 
	};
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

Var* mse(Var* a, Var* b, Var* out)
{
	auto c = sub(a, b, nullptr);
	return pow(c, 2.0f, out);
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
			inc(c, &t->data[0]);
		}
		return t;
	}

	static Tensor* vec_mat_mul(Tensor* vec, Tensor* mat)
	{
		int rows = mat->shape[0];
		int cols = mat->shape[1];

		auto t = new Tensor({ rows });
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				auto c = mul(&vec->data[j], &mat->data[i * cols + j], nullptr);
				inc(c, &t->data[i]);
			}
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

	static Tensor* mse_loss(Tensor* p, Tensor* y)
	{
		auto t = new Tensor(p->shape);
		for (int i = 0; i < p->size(); i++)
		{
			mse(&p->data[i], &y->data[i], &t->data[i]);
		}
		return t;
	}
};


int main(int argc, char **argv)
{

	auto x = new Tensor({ 8 }, 0, 1.0f);
	auto y = new Tensor({ 1 }, 1.0f);

	auto w1 = new Tensor({ 8, 8 }, 0, 1.0f);
	auto w2 = new Tensor({ 8, 8 }, 0, 1.0f);
	auto w3 = new Tensor({ 4, 4 }, 0, 1.0f);
	auto w4 = new Tensor({ 1, 4 }, 0, 1.0f);
	auto w5 = new Tensor({ 1 }, 0, 1.0f);

	std::vector<Tensor*> weights;
	weights.push_back(w1);
	weights.push_back(w2);
	weights.push_back(w3);
	weights.push_back(w4);
	weights.push_back(w5);

	auto eval = [&]() -> Tensor*
	{
		auto z1 = Tensor::vec_mat_mul(x,  w1);
		auto h1 = Tensor::sigmoid(z1);
		auto z2 = Tensor::vec_mat_mul(h1, w2);
		auto h2 = Tensor::sigmoid(z2);
		auto z3 = Tensor::vec_mat_mul(h2, w3);
		auto h3 = Tensor::sigmoid(z3);
		auto z4 = Tensor::vec_mat_mul(h3, w4);
		auto h4 = Tensor::sigmoid(z4);
		auto z5 = Tensor::dot(h4, w5);
		auto h5 = Tensor::sigmoid(z5);
		auto loss = Tensor::mse_loss(h5, y);
		return loss;
	};

	auto loss = eval();

	loss->data[0].derive();

	// Gradient Check
	{
		float agg_ana_grad = 0.0f;
		float agg_num_grad = 0.0f;
		float agg_grad_diff = 0.0f;

		for (auto w : weights)
		{
			for (int i = 0; i < w->size(); i++)
			{
				auto ow = w->data[i].v;

				w->data[i].v = ow - EPSILON;
				auto l_loss = eval();

				w->data[i].v = ow + EPSILON;
				auto r_loss = eval();

				float num_grad = (r_loss->data[0].v - l_loss->data[0].v) / (2.0f * EPSILON);
				float ana_grad = w->data[i].dv;

				printf("NUM: %f\n", num_grad);
				printf("ANA: %f\n\n", ana_grad);

				agg_ana_grad += (ana_grad * ana_grad);
				agg_num_grad += (num_grad * num_grad);
				agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

				w->data[i].v = ow;
			}
		}

		if ((agg_grad_diff) == 0.0f && (agg_ana_grad + agg_num_grad) == 0.0f)
		{
			printf("GRADIENT CHECK RESULT: %f\n", 0.0f);
		}
		else
		{
			printf("GRADIENT CHECK RESULT: %f\n", (agg_grad_diff) / (agg_ana_grad + agg_num_grad));

			if ((agg_grad_diff) / (agg_ana_grad + agg_num_grad) > EPSILON)
			{
				printf("MODEL GRADIENTS VALIDATION FAILED");
			}
		}
	}

	return 0;
}