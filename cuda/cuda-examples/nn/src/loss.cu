#include "loss.cuh"

using namespace loss;

__device__ float d_mse_evaluate(float p_val, float y_val)
{
    return ((p_val - y_val) * (p_val - y_val));
}

__device__ float d_mse_derive(float p_val, float y_val)
{
    return 2.0f * (p_val - y_val);
}

__global__ void k_mse_evaluate(float *p, float *y, int cnt, float *out_val)
{
    __shared__ float temp[THREADS_PER_BLOCK];
    memset(temp, 0, THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        temp[threadIdx.x] = d_mse_evaluate(p[tid], y[tid]);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0.0f;

        for (int i = 0; i < THREADS_PER_BLOCK; i++)
        {
            sum += temp[i];
        }

        atomicAdd(out_val, sum);
    }
}

__global__ void k_mse_derive(float *p, float *y, float *out, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        out[tid] = d_mse_derive(p[tid], y[tid]);
    }
}

void MeanSquaredError::evaluate(NdArray *p, NdArray *y, float *d_out_val)
{
    k_mse_evaluate<<<p->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(p->data(), y->data(), p->count(), d_out_val);
}

NdArray *MeanSquaredError::derive(NdArray *p, NdArray *y)
{
    NdArray *dl = new NdArray(true, p->dims());

    k_mse_derive<<<p->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(p->data(), y->data(), dl->data(), p->count());

    return dl;
}