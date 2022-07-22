#include <util.cuh>
#include <ndarray.cuh>

#define THREADS_PER_BLOCK 32

#define BATCH_SIZE 512
#define INPUT_SIZE 512
#define OUTPUT_SIZE 2048

#define EPOCHS 10

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

void set_test_vals(float *A, float *B)
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

__global__ void k_matmul_1(float *in, float *w, float *out,
						   int in_col_cnt, int out_col_cnt, int out_elem_cnt)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int out_elem_idx = tid;

	if (out_elem_idx < out_elem_cnt)
	{
		int in_row_idx = out_elem_idx / out_col_cnt;
		int w_col_idx = out_elem_idx % out_col_cnt;

		for (int in_col_idx = 0; in_col_idx < in_col_cnt; in_col_idx++)
		{
			out[out_elem_idx] += (in[in_row_idx * in_col_cnt + in_col_idx] * w[w_col_idx + (in_col_idx * out_col_cnt)]);
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

__global__ void k_matmul_3(float *in, float *w, float *out,
						   int in_col_cnt, int out_row_cnt, int out_col_cnt)
{
	int out_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int out_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

	int out_elem_idx = out_row_idx * out_col_cnt + out_col_idx;

	if (out_col_idx < out_col_cnt && out_row_idx < out_row_cnt)
	{
		int in_row_idx = out_row_idx;
		int w_col_idx = out_col_idx;

		for (int in_col_idx = 0; in_col_idx < in_col_cnt; in_col_idx++)
		{
			int w_row_idx = in_col_idx;
			out[out_elem_idx] += (in[in_row_idx * in_col_cnt + in_col_idx] * w[w_row_idx * out_col_cnt + w_col_idx]);
		}
	}
}

__global__ void k_activate_1(float *out)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int out_elem_idx = tid;

	if (tid < (BATCH_SIZE * OUTPUT_SIZE))
	{
		out[out_elem_idx] = (12.0f * 89.0f - 666);
	}
}

__global__ void k_activate_2(float *out)
{
	int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

	int out_elem_idx = row_idx * OUTPUT_SIZE + col_idx;

	if (col_idx < OUTPUT_SIZE && row_idx < BATCH_SIZE)
	{
		out[out_elem_idx] = (12.0f * 89.0f - 666);
	}
}

void compare_matricies(ArrayNd *A, ArrayNd *B)
{
	A->to_cpu();
	B->to_cpu();

	for (int i = 0; i < 10; i++)
	{
		int idx = rand() % (BATCH_SIZE * OUTPUT_SIZE);
		printf("%d\t%f\t%f\n", idx, A->data()[idx], B->data()[idx]);
	}

	float diff = 0.0f;

	for (int i = 0; i < A->count(); i++)
	{
		diff += abs(A->data()[i] - B->data()[i]);
	}

	printf("\nDIFF: %f\n", diff);
}

void matmul_perf_test()
{
	CudaStopWatch gsw;

	Array2d *C2 = new Array2d(true, BATCH_SIZE, OUTPUT_SIZE);
	Array2d *C4 = new Array2d(true, BATCH_SIZE, OUTPUT_SIZE);

	Array2d *A = new Array2d(false, BATCH_SIZE, INPUT_SIZE);
	Array2d *B = new Array2d(false, INPUT_SIZE, OUTPUT_SIZE);

	A->rands(0.0f, 1.0f);
	B->rands(0.0f, 1.0f);

	// set_test_vals(A->data(), B->data());

	A->to_cuda();
	B->to_cuda();

	{
		gsw.start();

		for (int i = 0; i < EPOCHS; i++)
		{
			C2->zeros();
			k_matmul_1<<<((BATCH_SIZE * OUTPUT_SIZE) / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(A->data(), B->data(), C2->data(),
																									INPUT_SIZE, OUTPUT_SIZE, (BATCH_SIZE * OUTPUT_SIZE));
		}

		gsw.stop();
		gsw.print_elapsed_seconds();
	}

	{
		gsw.start();

		for (int i = 0; i < EPOCHS; i++)
		{
			C4->zeros();
			unsigned int grid_rows = (BATCH_SIZE / THREADS_PER_BLOCK) + 1;
			unsigned int grid_cols = (OUTPUT_SIZE / THREADS_PER_BLOCK) + 1;
			dim3 dimGrid(grid_cols, grid_rows);
			dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
			k_matmul_3<<<dimGrid, dimBlock>>>(A->data(), B->data(), C4->data(),
											  INPUT_SIZE, BATCH_SIZE, OUTPUT_SIZE);
		}

		gsw.stop();
		gsw.print_elapsed_seconds();
	}

	printf("\n");

	compare_matricies(C2, C4);
	delete C2;
	delete C4;

	delete A;
	delete B;
}

void activate_perf_test()
{

	CudaStopWatch gsw;

	Array2d *C2 = new Array2d(true, BATCH_SIZE, OUTPUT_SIZE);
	Array2d *C4 = new Array2d(true, BATCH_SIZE, OUTPUT_SIZE);

	{
		gsw.start();

		for (int i = 0; i < EPOCHS; i++)
		{
			C2->zeros();
			k_activate_1<<<((BATCH_SIZE * OUTPUT_SIZE) / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(C2->data());
		}

		gsw.stop();
		gsw.print_elapsed_seconds();
	}

	{
		gsw.start();

		for (int i = 0; i < EPOCHS; i++)
		{
			C4->zeros();
			unsigned int grid_rows = (BATCH_SIZE / THREADS_PER_BLOCK) + 1;
			unsigned int grid_cols = (OUTPUT_SIZE / THREADS_PER_BLOCK) + 1;
			dim3 dimGrid(grid_cols, grid_rows);
			dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
			k_activate_2<<<dimGrid, dimBlock>>>(C4->data());
		}

		gsw.stop();
		gsw.print_elapsed_seconds();
	}

	delete C2;
	delete C4;
	
}

int main(int argc, char **argv)
{
	matmul_perf_test();

	activate_perf_test();

	return 0;
}