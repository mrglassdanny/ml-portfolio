#include <util.cuh>
#include <tensor.cuh>

#define THREADS_PER_BLOCK 32

#define BATCH_SIZE 1024
#define INPUT_SIZE 1024
#define OUTPUT_SIZE 512

/*
	Example matrix multiplication:

			[ 1 2 1 ]
			[ 3 1 2 ]
			[ 3 3 1 ]

			[ 1 2 ]
			[ 2 2 ]
			[ 1 3 ]

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

	C2->to_cpu();

	for (int i = 0; i < 1000; i++)
	{
		printf("C1[%d] = %f\tC2[%d] = %f\n", i, C1->get_data()[i], i, C2->get_data()[i]);
	}

	delete C1;
	delete C2;

	delete A;
	delete B;

	return 0;
}