#include "activation.cuh"

using namespace nn::layer;

__device__ float d_sigmoid_evaluate(float val)
{
    return (1.0f / (1.0f + exp(-val)));
}

__device__ float d_sigmoid_derive(float sigmoid_val)
{
    return (sigmoid_val) * (1.0f - sigmoid_val);
}

__global__ void k_sigmoid_evaluate(float *in, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;
        in[elem_idx] = d_sigmoid_evaluate(in[elem_idx]);
    }
}

__global__ void k_sigmoid_derive(float *in, float *n, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;
        in[elem_idx] *= d_sigmoid_derive(n[elem_idx]);
    }
}

__device__ float d_tanh_evaluate(float val)
{
    return ((exp(val) - exp(-val)) / (exp(val) + exp(-val)));
}

__device__ float d_tanh_derive(float tanh_val)
{
    return (1.0f - (tanh_val * tanh_val));
}

__global__ void k_tanh_evaluate(float *in, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;
        in[elem_idx] = d_tanh_evaluate(in[elem_idx]);
    }
}

__global__ void k_tanh_derive(float *in, float *n, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;
        in[elem_idx] *= d_tanh_derive(n[elem_idx]);
    }
}

__device__ float d_relu_evaluate(float val)
{
    return val > 0.0f ? val : 0.0f;
}

__device__ float d_relu_derive(float relu_val)
{
    return relu_val > 0.0f ? 1.0f : 0.0f;
}

__global__ void k_relu_evaluate(float *in, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;
        in[elem_idx] = d_relu_evaluate(in[elem_idx]);
    }
}

__global__ void k_relu_derive(float *in, float *n, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;
        in[elem_idx] *= d_relu_derive(n[elem_idx]);
    }
}

void Activation::evaluate(Tensor *in, int batch_size, int cnt, ActivationType activation)
{
    int grid_row_cnt = (batch_size / CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (cnt / CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

    switch (activation)
    {
    case ActivationType::None:
        break;
    case ActivationType::Sigmoid:
        k_sigmoid_evaluate<<<grid_dims, block_dims>>>(in->data(), batch_size, cnt);
        break;
    case ActivationType::Tanh:
        k_tanh_evaluate<<<grid_dims, block_dims>>>(in->data(), batch_size, cnt);
        break;
    case ActivationType::ReLU:
        k_relu_evaluate<<<grid_dims, block_dims>>>(in->data(), batch_size, cnt);
        break;
    default: // None
        break;
    }
}

void Activation::derive(Tensor *in, Tensor *n, int batch_size, int cnt, ActivationType activation)
{
    int grid_row_cnt = (batch_size / CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (cnt / CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

    switch (activation)
    {
    case ActivationType::None:
        break;
    case ActivationType::Sigmoid:
        k_sigmoid_derive<<<grid_dims, block_dims>>>(in->data(), n->data(), batch_size, cnt);
        break;
    case ActivationType::Tanh:
        k_tanh_derive<<<grid_dims, block_dims>>>(in->data(), n->data(), batch_size, cnt);
        break;
    case ActivationType::ReLU:
        k_relu_derive<<<grid_dims, block_dims>>>(in->data(), n->data(), batch_size, cnt);
        break;
    default: // None
        break;
    }
}

void Activation::summarize(ActivationType activation)
{
    printf("Activation: ");

    switch (activation)
    {
    case ActivationType::None:
        printf("None");
        break;
    case ActivationType::Sigmoid:
        printf("Sigmoid");
        break;
    case ActivationType::Tanh:
        printf("Tanh");
        break;
    case ActivationType::ReLU:
        printf("ReLU");
        break;
    default: // None
        break;
    }
}