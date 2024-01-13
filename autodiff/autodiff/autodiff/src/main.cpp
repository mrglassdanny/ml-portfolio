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
	Sub,
	Mul,
	Div,
	Pow,
	Sigmoid
};

struct Var
{
	float v = 0.0f;
	int i = -1;

	Var() {}

	Var(float v)
	{
		this->v = v;
	}

	Var(float v, int i)
	{
		this->v = v;
		this->i = i;
	}

	void print()
	{
		printf("%f (%d): ", this->v, this->i);
	}
};

struct Node
{
	Oper op;
	int as[2];
	float pds[2];

	Node() {}

	Node(Oper op, int a1, int a2, float pd1, float pd2)
	{
		this->op = op;
		this->as[0] = a1;
		this->as[1] = a2;
		this->pds[0] = pd1;
		this->pds[1] = pd2;
	}
};

class Tensor
{
public:
	std::vector<int> shape;
	Var *data;

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

	void print()
	{
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
};

class Context
{
public:
	std::vector<Node> tape;

	Context() {}

	Var var(float v)
	{
		auto var = Var(v, this->tape.size());
		this->tape.push_back(Node(None, var.i, var.i, 0.0f, 0.0f));
		return var;
	}

	Tensor *tensor(Tensor *tensor)
	{
		for (int i = 0; i < tensor->size(); i++)
		{
			int idx = this->tape.size();
			tensor->data[i].i = idx;
			this->tape.push_back(Node(None, idx, idx, 0.0f, 0.0f));
		}
		return tensor;
	}

	Var op(float v, Oper op, int a1, int a2, float pd1, float pd2)
	{
		Var c;
		c.v = v;
		c.i = this->tape.size();
		this->tape.push_back(Node(op, a1, a2, pd1, pd2));
		return c;
	}

	Tensor *backward()
	{
		auto grad = new Tensor({(int)this->tape.size()}, 0.0f);
		grad->data[grad->size() - 1].v = 1.0f;
		for (int i = this->tape.size() - 1; i >= 0; i--)
		{
			grad->data[this->tape[i].as[0]].v += this->tape[i].pds[0] * grad->data[i].v;
			grad->data[this->tape[i].as[1]].v += this->tape[i].pds[1] * grad->data[i].v;
		}
		return grad;
	}
};

Var add(Context *ctx, Var a, Var b)
{
	Var c = ctx->op(a.v + b.v, Add, a.i, b.i, 1.0f, 1.0f);
	return c;
}

Var sub(Context *ctx, Var a, Var b)
{
	Var c = ctx->op(a.v - b.v, Sub, a.i, b.i, 1.0f, -1.0f);
	return c;
}

Var mul(Context *ctx, Var a, Var b)
{
	Var c = ctx->op(a.v * b.v, Mul, a.i, b.i, b.v, a.v);
	return c;
}

Var pow(Context *ctx, Var a, float b)
{
	Var c = ctx->op(pow(a.v, b), Pow, a.i, -1, b * pow(a.v, b - 1), 0.0f);
	return c;
}

Var sig(Context *ctx, Var a)
{
	float v = (1.0f / (1.0f + exp(-a.v)));
	Var c = ctx->op(v, Sigmoid, a.i, -1, (v * (1.0f - v)), 0.0f);
	return c;
}

Var mse(Context *ctx, Var a, Var b)
{
	auto c = sub(ctx, a, b);
	return pow(ctx, c, 2.0f);
}

Tensor *dot(Context *ctx, Tensor *a, Tensor *b)
{
	auto t = ctx->tensor(new Tensor({1}));
	for (int i = 0; i < a->size(); i++)
	{
		auto c = mul(ctx, a->data[i], b->data[i]);
		t->data[i] = add(ctx, c, t->data[i]);
	}
	return t;
}

Tensor *vec_mat_mul(Context *ctx, Tensor *vec, Tensor *mat)
{
	int rows = mat->shape[0];
	int cols = mat->shape[1];

	auto t = new Tensor({rows});
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			auto c = mul(ctx, vec->data[j], mat->data[i * cols + j]);
			t->data[i] = add(ctx, c, t->data[i]);
		}
	}
	return t;
}

Tensor *sigmoid(Context *ctx, Tensor *a)
{
	auto t = ctx->tensor(new Tensor(a->shape));
	for (int i = 0; i < a->size(); i++)
	{
		t->data[i] = sig(ctx, a->data[i]);
	}
	return t;
}

Tensor *mse_loss(Context *ctx, Tensor *p, Tensor *y)
{
	auto t = ctx->tensor(new Tensor(p->shape));
	for (int i = 0; i < p->size(); i++)
	{
		t->data[i] = mse(ctx, p->data[i], y->data[i]);
	}
	return t;
}

int main(int argc, char **argv)
{
	auto ctx = Context();

	auto x = ctx.tensor(new Tensor({8}, 0, 1.0f));
	auto y = ctx.tensor(new Tensor({1}, 1.0f));

	auto w1 = ctx.tensor(new Tensor({8, 8}, 0, 1.0f));
	auto w2 = ctx.tensor(new Tensor({8, 8}, 0, 1.0f));
	auto w3 = ctx.tensor(new Tensor({4, 4}, 0, 1.0f));
	auto w4 = ctx.tensor(new Tensor({1, 4}, 0, 1.0f));
	auto w5 = ctx.tensor(new Tensor({1}, 0, 1.0f));

	std::vector<Tensor *> weights;
	weights.push_back(w1);
	weights.push_back(w2);
	weights.push_back(w3);
	weights.push_back(w4);
	weights.push_back(w5);

	auto eval = [&]() -> Tensor *
	{
		auto z1 = vec_mat_mul(&ctx, x, w1);
		auto h1 = sigmoid(&ctx, z1);
		auto z2 = vec_mat_mul(&ctx, h1, w2);
		auto h2 = sigmoid(&ctx, z2);
		auto z3 = vec_mat_mul(&ctx, h2, w3);
		auto h3 = sigmoid(&ctx, z3);
		auto z4 = vec_mat_mul(&ctx, h3, w4);
		auto h4 = sigmoid(&ctx, z4);
		auto z5 = dot(&ctx, h4, w5);
		auto h5 = sigmoid(&ctx, z5);
		auto loss = mse_loss(&ctx, h5, y);
		return loss;
	};

	auto loss = eval();

	auto grad = ctx.backward();

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
				float ana_grad = grad->data[w->data[i].i].v;

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