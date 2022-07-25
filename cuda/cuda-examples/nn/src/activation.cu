#include "activation.cuh"

using namespace activation;

__device__ float d_sigmoid_evaluate(float val)
{
    return (1.0f / (1.0f + exp(-val)));
}

__device__ float d_sigmoid_derive(float sigmoid_val)
{
    return (sigmoid_val) * (1.0f - sigmoid_val);
}

__global__ void k_sigmoid_evaluate(float *in, float *out, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        out[tid] = d_sigmoid_evaluate(in[tid]);
    }
}

__global__ void k_sigmoid_derive(float *in, float *out, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        out[tid] = d_sigmoid_derive(in[tid]);
    }
}

void Sigmoid::evaluate(NdArray *in, NdArray *out)
{
    k_sigmoid_evaluate<<<in->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(in->data(), out->data(), in->count());
}

void Sigmoid::derive(NdArray *in, NdArray *out)
{

    k_sigmoid_derive<<<in->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(in->data(), out->data(), in->count());
}
