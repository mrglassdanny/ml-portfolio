#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

namespace epyon
{
	namespace core
	{
		enum Oper
		{
			None = 0,
			Add,
			Mul,
			Exp
		};

		struct Var
		{
			float v = 0.0f;
			float dv = 0.0f;
			Oper oper = None;
			void (*df)(Var *in, Var **out, int cnt);
			Var **prev;
			int cur = 0;
			int cnt = 0;
		};

		void add(Var *in_a, Var *in_b, Var *out)
		{
			out->v = in_a->v + in_b->v;
			out->prev[out->cur++] = in_a;
			out->prev[out->cur++] = in_b;
			out->df = derive_add;
			out->oper = Add;
		}

		void add(Var *in, Var *out)
		{
			out->v += in->v;
			out->prev[out->cur++] = in;
			out->df = derive_add;
			out->oper = Add;
		}

		void derive_add(Var *in, Var **prev, int cnt)
		{
			for (int i = 0; i < cnt; i++)
			{
				prev[i]->dv += in->dv;
			}

			in->cur = 0;
		}

		__device__ void d_add(Var *in_a, Var *in_b, Var *out)
		{
			out->v = in_a->v + in_b->v;
			out->prev[out->cur++] = in_a;
			out->prev[out->cur++] = in_b;
			out->df = d_derive_add;
			out->oper = Add;
		}

		__device__ void d_add(Var *in, Var *out)
		{
			out->v += in->v;
			out->prev[out->cur++] = in;
			out->df = d_derive_add;
			out->oper = Add;
		}

		__device__ void d_derive_add(Var *in, Var **prev, int cnt)
		{
			for (int i = 0; i < cnt; i++)
			{
				prev[i]->dv += in->dv;
			}

			in->cur = 0;
		}
	}
}