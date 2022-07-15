#include <util.cuh>
#include <tensor.cuh>

#define THREADS_PER_BLOCK 32

#define BATCH_SIZE 16
#define INPUT_SIZE 48
#define OUTPUT_SIZE 12

/*
	Example matrix multiplication:

			[ 1 2 1 ]
			[ 3 1 2 ]
			[ 3 3 1 ]

			x

			[ 1 2 ]
			[ 2 2 ]
			[ 1 3 ]

			=

			[  6  9  ]
			[  7 14  ]
			[ 10 15  ]
*/

void set_vals(float *A, float *B)
{
	A[0] = 1;
	A[1] = 2;
	A[2] = 1;
	A[3] = 3;
	A[4] = 1;
	A[5] = 2;
	A[6] = 3;
	A[7] = 3;
	A[8] = 1;

	B[0] = 1;
	B[1] = 2;
	B[2] = 2;
	B[3] = 2;
	B[4] = 1;
	B[5] = 3;
}

void matmul(float *A, float *B, float *C)
{
	for (int i = 0; i < BATCH_SIZE; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			for (int k = 0; k < OUTPUT_SIZE; k++)
			{
				C[i * OUTPUT_SIZE + k] += (A[i * INPUT_SIZE + j] * B[j * OUTPUT_SIZE + k]);
			}
		}
	}
}

__global__ void k_matmul_1(float *A, float *B, float *C)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < (BATCH_SIZE * OUTPUT_SIZE))
	{
		int A_idx = tid / OUTPUT_SIZE;
		int B_idx = tid % OUTPUT_SIZE;

		for (int i = 0; i < INPUT_SIZE; i++)
		{
			C[tid] += (A[A_idx * INPUT_SIZE + i] * B[B_idx + (i * OUTPUT_SIZE)]);
		}
	}
}

__global__ void k_matmul_2(float *A, float *B, float *C)
{
	__shared__ float temp[THREADS_PER_BLOCK];
	memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int batch_idx = tid / (INPUT_SIZE * OUTPUT_SIZE);
	int A_idx = (tid % INPUT_SIZE) + (batch_idx * INPUT_SIZE);
	int B_idx = (tid % INPUT_SIZE) * OUTPUT_SIZE + ((tid - (batch_idx * (INPUT_SIZE * OUTPUT_SIZE))) / INPUT_SIZE);

	if (tid < (BATCH_SIZE * INPUT_SIZE * OUTPUT_SIZE))
	{
		temp[threadIdx.x] = A[A_idx] * B[B_idx];
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		int lower_idx = tid / INPUT_SIZE;
		int upper_idx = ((tid + THREADS_PER_BLOCK) / INPUT_SIZE);

		if (INPUT_SIZE >= THREADS_PER_BLOCK)
		{
			if (lower_idx == upper_idx)
			{
				float sum = 0.0f;

				for (int i = 0; i < THREADS_PER_BLOCK; i++)
				{
					sum += temp[i];
				}

				atomicAdd(&C[lower_idx], sum);
			}
			else
			{
				float sums[2] = {0.0f, 0.0f};

				for (int i = 0; i < THREADS_PER_BLOCK; i++)
				{
					int idx = ((tid + i) / INPUT_SIZE);
					if (idx == lower_idx)
					{
						sums[0] += temp[i];
					}
					else
					{
						sums[1] += temp[i];
					}
				}

				atomicAdd(&C[lower_idx], sums[0]);
				if (upper_idx < (BATCH_SIZE * OUTPUT_SIZE))
				{
					atomicAdd(&C[upper_idx], sums[1]);
				}
			}
		}
		else
		{
			for (int i = 0; i < THREADS_PER_BLOCK; i++)
			{
				int idx = ((tid + i) / INPUT_SIZE);
				if (idx < (BATCH_SIZE * OUTPUT_SIZE))
				{
					atomicAdd(&C[idx], temp[i]);
				}
			}
		}
	}
}

int main(int argc, char **argv)
{
	StopWatch sw;

	Tensor *C1 = new Tensor(false, Dimensions(BATCH_SIZE, OUTPUT_SIZE));
	Tensor *C2 = new Tensor(true, Dimensions(BATCH_SIZE, OUTPUT_SIZE));
	Tensor *C3 = new Tensor(true, Dimensions(BATCH_SIZE, OUTPUT_SIZE));

	Tensor *A = new Tensor(false, Dimensions(BATCH_SIZE, INPUT_SIZE));
	Tensor *B = new Tensor(false, Dimensions(INPUT_SIZE, OUTPUT_SIZE));

	A->rands(0.0f, 1.0f);
	B->rands(0.0f, 1.0f);

	// set_vals(A->get_data(), B->get_data());

	// A->print();
	// B->print();

	{
		C1->zeros();

		sw.start();

		matmul(A->get_data(), B->get_data(), C1->get_data());

		printf("\n");

		sw.stop();
		sw.print_elapsed_seconds();
	}

	{
		C2->zeros();

		A->to_cuda();
		B->to_cuda();

		sw.start();

		k_matmul_1<<<((BATCH_SIZE * OUTPUT_SIZE) / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(A->get_data(), B->get_data(), C2->get_data());

		printf("\n");

		sw.stop();
		sw.print_elapsed_seconds();
	}

	{
		C3->zeros();

		A->to_cuda();
		B->to_cuda();

		sw.start();

		k_matmul_2<<<((BATCH_SIZE * INPUT_SIZE * OUTPUT_SIZE) / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(A->get_data(), B->get_data(), C3->get_data());

		printf("\n");

		sw.stop();
		sw.print_elapsed_seconds();
	}

	printf("\n");

	C2->to_cpu();
	C3->to_cpu();

	C1->print();
	C2->print();
	C3->print();

	delete C1;
	delete C2;
	delete C3;

	delete A;
	delete B;

	return 0;
}