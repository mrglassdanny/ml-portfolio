#include "loss.cuh"

using namespace zero::core;
using namespace zero::nn::loss;

__device__ float d_mse_evaluate(float p_val, float y_val)
{
    return ((p_val - y_val) * (p_val - y_val));
}

__device__ float d_mse_derive(float p_val, float y_val)
{
    return 2.0f * (p_val - y_val);
}

__global__ void k_mse_evaluate(float *p, float *y, float *out, int row_cnt, int col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        out[row_idx * col_cnt + col_idx] = d_mse_evaluate(p[row_idx * col_cnt + col_idx], y[row_idx * col_cnt + col_idx]);
    }
}

__global__ void k_mse_derive(float *p, float *y, float *out, int row_cnt, int col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        out[row_idx * col_cnt + col_idx] = d_mse_derive(p[row_idx * col_cnt + col_idx], y[row_idx * col_cnt + col_idx]);
    }
}

__device__ float d_softmax(float val, float *arr, int cnt)
{
    float e_sum_val = 0.0f;

    for (int i = 0; i < cnt; i++)
    {
        e_sum_val += exp(arr[i]);
    }

    return exp(val) / e_sum_val;
}

__device__ float d_cross_entropy_evaluate(float p_val, float y_val, float *p, int cnt)
{
    float np_val = d_softmax(p_val, p, cnt);
    return -(y_val * log(np_val));
}

__device__ float d_cross_entropy_derive(float p_val, float y_val, float *p, int cnt)
{
    float np_val = d_softmax(p_val, p, cnt);
    return np_val - y_val;
}

__global__ void k_cross_entropy_evaluate(float *p, float *y, float *out, int row_cnt, int col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        out[row_idx * col_cnt + col_idx] = d_cross_entropy_evaluate(p[row_idx * col_cnt + col_idx], y[row_idx * col_cnt + col_idx],
                                                                    p, col_cnt);
    }
}

__global__ void k_cross_entropy_derive(float *p, float *y, float *out, int row_cnt, int col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        out[row_idx * col_cnt + col_idx] = d_cross_entropy_derive(p[row_idx * col_cnt + col_idx], y[row_idx * col_cnt + col_idx],
                                                                  p, col_cnt);
    }
}

void Loss::summarize()
{
    std::string cls_name(typeid(*this).name());
    printf("%s", cls_name.c_str());
}

void MSE::evaluate(Tensor *p, Tensor *y, Tensor *out)
{
    int batch_size = p->shape()[0];
    int output_cnt = p->shape()[1];

    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (output_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_mse_evaluate<<<grid_dims, block_dims>>>(p->data(), y->data(), out->data(), batch_size, output_cnt);
}

Tensor *MSE::derive(Tensor *p, Tensor *y)
{
    Tensor *out = new Tensor(true, p->shape());

    int batch_size = p->shape()[0];
    int output_cnt = p->shape()[1];

    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (output_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_mse_derive<<<grid_dims, block_dims>>>(p->data(), y->data(), out->data(), batch_size, output_cnt);

    return out;
}

Loss *MSE::copy()
{
    return new MSE();
}

void CrossEntropy::evaluate(Tensor *p, Tensor *y, Tensor *out)
{
    int batch_size = p->shape()[0];
    int output_cnt = p->shape()[1];

    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (output_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_cross_entropy_evaluate<<<grid_dims, block_dims>>>(p->data(), y->data(), out->data(), batch_size, output_cnt);
}

Tensor *CrossEntropy::derive(Tensor *p, Tensor *y)
{
    Tensor *out = new Tensor(true, p->shape());

    int batch_size = p->shape()[0];
    int output_cnt = p->shape()[1];

    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (output_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_cross_entropy_derive<<<grid_dims, block_dims>>>(p->data(), y->data(), out->data(), batch_size, output_cnt);

    return out;
}

Loss *CrossEntropy::copy()
{
    return new CrossEntropy();
}