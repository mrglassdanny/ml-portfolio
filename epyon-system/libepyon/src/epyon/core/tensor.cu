#include "tensor.cuh"

namespace epyon
{
	namespace core
	{
		__host__ __device__ void add(Var *in_a, Var *in_b, Var *out)
		{
			out->v = in_a->v + in_b->v;
			out->prev[out->cur++] = in_a;
			out->prev[out->cur++] = in_b;
			out->df = derive_add;
			out->oper = Add;
		}

		__host__ __device__ void add(Var *in, Var *out)
		{
			out->v += in->v;
			out->prev[out->cur++] = in;
			out->df = derive_add;
			out->oper = Add;
		}

		__host__ __device__ void derive_add(Var *in, Var **prev, int cnt)
		{
			for (int i = 0; i < cnt; i++)
			{
				prev[i]->dv += in->dv;
			}

			in->cur = 0;
		}
	}
}
