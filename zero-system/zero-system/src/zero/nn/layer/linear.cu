#include "linear.cuh"

using namespace zero::core;
using namespace zero::nn::layer;

__global__ void k_linear_evaluate(float *in, float *w, float *b, float *out,
                                  int batch_size, int in_cnt, int out_cnt)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx < out_cnt && batch_idx < batch_size)
    {
        int out_elem_idx = batch_idx * out_cnt + out_idx;
        int w_col_idx = out_idx;

        for (int in_idx = 0; in_idx < in_cnt; in_idx++)
        {
            int w_row_idx = in_idx;
            out[out_elem_idx] += (in[batch_idx * in_cnt + in_idx] * w[w_row_idx * out_cnt + w_col_idx]);
        }

        out[out_elem_idx] += b[w_col_idx];
    }
}

__global__ void k_linear_inc_param_derivatives(float *in, float *in_n, float *n, float *dw, float *db,
                                               int batch_size, int in_cnt, int n_cnt, int w_row_cnt, int w_col_cnt)
{
    int w_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int w_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_col_idx < w_col_cnt && w_row_idx < w_row_cnt)
    {
        int w_elem_idx = w_row_idx * w_col_cnt + w_col_idx;
        int n_idx = w_row_idx;
        int in_idx = w_col_idx;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            dw[w_elem_idx] += (in[batch_idx * in_cnt + in_idx] * n[batch_idx * n_cnt + n_idx]);

            if (w_row_idx == 0)
            {
                int b_elem_idx = w_col_idx;
                db[b_elem_idx] += in[batch_idx * in_cnt + in_idx];
            }
        }
    }
}

__global__ void k_linear_agg_derivatives(float *in, float *w, float *out, int batch_size, int in_cnt, int w_row_cnt, int w_col_cnt, int out_cnt)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx < out_cnt && batch_idx < batch_size)
    {
        int out_elem_idx = batch_idx * w_row_cnt + out_idx;
        int w_row_idx = out_idx;

        for (int in_idx = 0; in_idx < in_cnt; in_idx++)
        {
            int w_col_idx = in_idx;
            out[out_elem_idx] += (in[batch_idx * in_cnt + in_idx] * w[w_row_idx * w_col_cnt + w_col_idx]);
        }
    }
}

Linear::Linear(bool shared_params, Shape in_shape, Shape out_shape, Activation *activation, Initializer *initializer)
    : Learnable(shared_params)
{
    this->n_ = new Tensor(true, in_shape);
    this->dn_ = Tensor::zeros(true, in_shape);
    this->activation_ = activation;

    int in_cnt = (in_shape.dims_size() / this->batch_size());
    int out_cnt = (out_shape.dims_size() / this->batch_size());

    if (!this->shared_params_)
    {
        this->params_ = new Parameters(Shape(in_cnt, out_cnt), Shape(out_cnt), in_cnt, out_cnt, initializer);
    }
}

void Linear::evaluate(Tensor *out)
{
    int grid_row_cnt = (this->batch_size() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->out_features() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    Tensor *n = this->n_;
    Tensor *w = this->params_->weights();
    Tensor *b = this->params_->biases();

    k_linear_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), b->data(), out->data(),
                                                 this->batch_size(), this->in_features(), this->out_features());

    if (this->activation_ != nullptr)
    {
        this->activation_->evaluate(out, this->batch_size(), this->out_features());
    }
}

void Linear::derive(Tensor *in, Tensor *in_n)
{
    Tensor *n = this->n_;
    Tensor *dn = this->dn_;
    Tensor *w = this->params_->weights();
    Tensor *b = this->params_->biases();
    Tensor *dw = this->params_->weight_gradients();
    Tensor *db = this->params_->bias_gradients();

    if (this->activation_ != nullptr)
    {
        this->activation_->derive(in, in_n, this->batch_size(), this->out_features());
    }

    {
        int grid_row_cnt = (this->weight_rows() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->weight_cols() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

        k_linear_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), in_n->data(), n->data(), dw->data(), db->data(),
                                                                  this->batch_size(), this->out_features(), this->in_features(),
                                                                  this->weight_rows(), this->weight_cols());
    }

    {
        int grid_row_cnt = (this->batch_size() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->in_features() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

        k_linear_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), dn->data(),
                                                            this->batch_size(), this->out_features(), this->weight_rows(), this->weight_cols(), this->in_features());
    }
}

Shape Linear::input_shape()
{
    return this->n_->shape();
}

Shape Linear::output_shape()
{
    return Shape(this->batch_size(), this->weight_cols());
}

Layer *Linear::copy()
{
    auto lyr = new Linear(true, this->input_shape(), this->output_shape(), this->activation_->copy(), nullptr);
    lyr->share_parameters(this->params_);
    return lyr;
}

void Linear::validate()
{
    if (this->input_shape().num_dims() < 2)
    {
        ZERO_CORE_THROW_ERROR("LINEAR LAYER VALIDATION FAILED: invalid input shape");
    }

    if (this->output_shape().num_dims() != 2)
    {
        ZERO_CORE_THROW_ERROR("LINEAR LAYER VALIDATION FAILED: invalid output shape");
    }
}

void Linear::summarize()
{
    Layer::summarize();
}

int Linear::weight_rows()
{
    return this->params_->weights()->shape()[0];
}

int Linear::weight_cols()
{
    return this->params_->weights()->shape()[1];
}
