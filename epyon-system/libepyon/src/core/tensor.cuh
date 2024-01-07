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
		struct Var
		{
			float v = 0.0f;
			float dv = 0.0f;
		};
	}
}