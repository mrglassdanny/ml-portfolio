#include "layer.cuh"

using namespace zero::core;
using namespace zero::nn::layer;

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

void Activation::summarize()
{
    std::string cls_name(typeid(*this).name());
    printf("%s", cls_name.c_str());
}

void Sigmoid::evaluate(Tensor *in, int batch_size, int cnt)
{
    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_sigmoid_evaluate<<<grid_dims, block_dims>>>(in->data(), batch_size, cnt);
}

void Sigmoid::derive(Tensor *in, Tensor *n, int batch_size, int cnt)
{
    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_sigmoid_derive<<<grid_dims, block_dims>>>(in->data(), n->data(), batch_size, cnt);
}

Activation *Sigmoid::copy()
{
    return new Sigmoid();
}

void Tanh::evaluate(Tensor *in, int batch_size, int cnt)
{
    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_tanh_evaluate<<<grid_dims, block_dims>>>(in->data(), batch_size, cnt);
}

void Tanh::derive(Tensor *in, Tensor *n, int batch_size, int cnt)
{
    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_tanh_derive<<<grid_dims, block_dims>>>(in->data(), n->data(), batch_size, cnt);
}

Activation *Tanh::copy()
{
    return new Tanh();
}

void ReLU::evaluate(Tensor *in, int batch_size, int cnt)
{
    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_relu_evaluate<<<grid_dims, block_dims>>>(in->data(), batch_size, cnt);
}

void ReLU::derive(Tensor *in, Tensor *n, int batch_size, int cnt)
{
    int grid_row_cnt = (batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_relu_derive<<<grid_dims, block_dims>>>(in->data(), n->data(), batch_size, cnt);
}

Activation *ReLU::copy()
{
    return new ReLU();
}

Layer::~Layer()
{
    delete this->n_;
    delete this->dn_;

    if (this->activation_ != nullptr)
    {
        delete this->activation_;
    }
}

void Layer::summarize()
{
    std::string cls_name(typeid(*this).name());
    for (int i = cls_name.size(); i < 26; i++)
    {
        cls_name.push_back(' ');
    }

    printf("%s\t", cls_name.c_str());
    this->input_shape().print_pad(16, true);
    printf(" -> ");
    this->output_shape().print_pad(16, false);

    printf("\tActivation: ");
    if (this->activation_ != nullptr)
    {
        this->activation_->summarize();
    }
    else
    {
        printf("None");
    }
}

int Layer::in_features()
{
    return this->input_shape().dims_size() / this->batch_size();
}

int Layer::out_features()
{
    return this->output_shape().dims_size() / this->batch_size();
}

int Layer::batch_size()
{
    return this->n_->shape()[0];
}

void Layer::change_batch_size(int batch_size)
{
    this->n_->change_dim(0, batch_size);
    this->dn_->change_dim(0, batch_size);
}

Tensor *Layer::neurons()
{
    return this->n_;
}

void Layer::copy_neurons(Tensor *n)
{
    this->n_->copy(n);
}

Tensor *Layer::neuron_gradients()
{
    return this->dn_;
}

void Layer::zero_grad()
{
    this->dn_->zeros();
}
