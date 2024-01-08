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

		__host__ __device__ void add(Var *in_a, Var *in_b, Var *out);

		__host__ __device__ void add(Var *in, Var *out);

		__host__ __device__ void derive_add(Var *in, Var **prev, int cnt);
	}
}