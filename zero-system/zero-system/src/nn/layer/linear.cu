#include "linear.cuh"

using namespace nn::layer;

__global__ void k_linear_evaluate(float* in, float* w, float* b, float* out,
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

__global__ void k_linear_inc_param_derivatives(float* in, float* n, float* dw, float* db,
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

__global__ void k_linear_agg_derivatives(float* in, float* w, float* out, int batch_size, int in_cnt, int w_col_cnt, int out_cnt)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx < out_cnt && batch_idx < batch_size)
    {
        int out_elem_idx = batch_idx * w_col_cnt + out_idx;
        int w_row_idx = out_idx;

        for (int in_idx = 0; in_idx < in_cnt; in_idx++)
        {
            int w_col_idx = in_idx;
            out[out_elem_idx] += (in[batch_idx * in_cnt + in_idx] * w[w_row_idx * w_col_cnt + w_col_idx]);
        }
    }
}



Linear::Linear(Shape in_shape, Shape out_shape)
{
    this->n_ = new NdArray(true, in_shape);

    int in_cnt = (in_shape.dims_size() / this->batch_size());
    int out_cnt = (out_shape.dims_size() / this->batch_size());

    this->params_ = new Parameters(Shape(in_cnt, out_cnt), Shape(out_cnt), in_cnt, out_cnt);
}

void Linear::evaluate(NdArray* out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->out_features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    NdArray* n = this->n_;
    NdArray* w = this->params_->weights();
    NdArray* b = this->params_->biases();

    k_linear_evaluate << <grid_dims, block_dims >> > (n->data(), w->data(), b->data(), out->data(),
        this->batch_size(), this->in_features(), this->out_features());
}

NdArray* Linear::derive(NdArray* in)
{
    NdArray* n = this->n_;
    NdArray* w = this->params_->weights();
    NdArray* b = this->params_->biases();
    NdArray* dw = this->params_->weight_gradients();
    NdArray* db = this->params_->bias_gradients();

    {
        int grid_row_cnt = (this->weight_rows() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->weight_cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_inc_param_derivatives << <grid_dims, block_dims >> > (in->data(), n->data(), dw->data(), db->data(),
            this->batch_size(), this->out_features(), this->in_features(),
            this->weight_rows(), this->weight_cols());
    }

    NdArray* out = NdArray::zeros(true, this->input_shape());

    {
        int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->in_features() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_agg_derivatives << <grid_dims, block_dims >> > (in->data(), w->data(), out->data(),
            this->batch_size(), this->out_features(), this->weight_cols(), this->in_features());
    }

    delete in;
    return out;
}

Shape Linear::input_shape()
{
    return this->n_->shape();
}

Shape Linear::output_shape()
{
    return Shape(this->batch_size(), this->params_->weights()->shape()[1]);
}

void Linear::validate() {}

int Linear::in_features()
{
    return this->weight_rows();
}

int Linear::out_features()
{
    return this->weight_cols();
}

int Linear::weight_rows()
{
    return this->params_->weights()->shape()[0];
}

int Linear::weight_cols()
{
    return this->params_->weights()->shape()[1];
}
