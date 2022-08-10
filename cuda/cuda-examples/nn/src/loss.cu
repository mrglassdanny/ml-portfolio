#include "loss.cuh"

using namespace nn::loss;

__device__ float d_mse_evaluate(float p_val, float y_val)
{
    return ((p_val - y_val) * (p_val - y_val));
}

__device__ float d_mse_derive(float p_val, float y_val)
{
    return 2.0f * (p_val - y_val);
}

__global__ void k_mse_evaluate(float *p, float *y, float *out, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        out[tid] = d_mse_evaluate(p[tid], y[tid]);
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

void Loss::summarize()
{
    std::string cls_name(typeid(*this).name());
    printf("%s", cls_name.c_str());
}

void MSE::evaluate(NdArray *p, NdArray *y, NdArray *out)
{
    k_mse_evaluate<<<p->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(p->data(), y->data(), out->data(), p->count());
}

NdArray *MSE::derive(NdArray *p, NdArray *y)
{
    NdArray *dl = new NdArray(true, p->shape());

    k_mse_derive<<<p->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(p->data(), y->data(), dl->data(), p->count());

    return dl;
}


