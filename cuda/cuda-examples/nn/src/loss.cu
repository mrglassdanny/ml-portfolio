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

Loss::Loss(std::vector<Layer *> layers)
{
    this->lyrs_ = layers;
}

NdArray *Loss::loss(NdArray *p, NdArray *y)
{
    p->to_cuda();
    y->to_cuda();

    NdArray *out = new NdArray(true, p->shape());
    out->zeros();

    this->evaluate(p, y, out);

    return out;
}

void Loss::backward(NdArray *p, NdArray *y)
{
    p->to_cuda();
    y->to_cuda();

    NdArray *dl = this->derive(p, y);

    int lst_lyr_idx = this->lyrs_.size() - 1;
    for (int i = lst_lyr_idx; i >= 0; i--)
    {
        Layer *lyr = this->lyrs_[i];
        dl = lyr->backward(dl);
    }

    delete dl;
}

MSE::MSE(std::vector<Layer *> layers)
    : Loss(layers)
{

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


