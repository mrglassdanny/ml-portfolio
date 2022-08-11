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

__global__ void k_sigmoid_evaluate(float *in, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = d_sigmoid_evaluate(in[elem_idx]);
    }
}

__global__ void k_sigmoid_derive(float *in, float *n, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = in[elem_idx] * d_sigmoid_derive(d_sigmoid_evaluate(n[elem_idx]));
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

__global__ void k_tanh_evaluate(float *in, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = d_tanh_evaluate(in[elem_idx]);
    }
}

__global__ void k_tanh_derive(float *in, float *n, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = in[elem_idx] * d_tanh_derive(d_tanh_evaluate(n[elem_idx]));
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

__global__ void k_relu_evaluate(float *in, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = d_relu_evaluate(in[elem_idx]);
    }
}

__global__ void k_relu_derive(float *in, float *n, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = in[elem_idx] * d_relu_derive(d_relu_evaluate(n[elem_idx]));
    }
}

Activation::Activation(Shape shape)
{
    this->n_ = new NdArray(true, shape);
}

Shape Activation::input_shape()
{
    return this->n_->shape();
}

Shape Activation::output_shape()
{
    return this->n_->shape();
}

void Activation::validate() {}

int Activation::features()
{
    return (this->n_->shape().dims_size() / this->batch_size());
}

Sigmoid::Sigmoid(Shape shape)
    : Activation(shape)
{
}

void Sigmoid::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_sigmoid_evaluate<<<grid_dims, block_dims>>>(this->n_->data(), out->data(), this->batch_size(), this->features());
}

NdArray *Sigmoid::derive(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->shape());

    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_sigmoid_derive<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), out->data(), this->batch_size(), this->features());

    delete in;
    return out;
}

Tanh::Tanh(Shape shape)
    : Activation(shape)
{
}

void Tanh::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_tanh_evaluate<<<grid_dims, block_dims>>>(this->n_->data(), out->data(), this->batch_size(), this->features());
}

NdArray *Tanh::derive(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->shape());

    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_tanh_derive<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), out->data(), this->batch_size(), this->features());

    delete in;
    return out;
}

ReLU::ReLU(Shape shape)
    : Activation(shape)
{
}

void ReLU::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_relu_evaluate<<<grid_dims, block_dims>>>(this->n_->data(), out->data(), this->batch_size(), this->features());
}

NdArray *ReLU::derive(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->shape());

    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_relu_derive<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), out->data(), this->batch_size(), this->features());

    delete in;
    return out;
}
