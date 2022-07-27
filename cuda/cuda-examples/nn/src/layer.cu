#include "layer.cuh"

using namespace layer;

__global__ void k_linear_matmul_w_bias(float *in, float *w, float *out, float *b,
                                       int in_col_cnt, int out_row_cnt, int out_col_cnt)
{
    int out_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_col_idx < out_col_cnt && out_row_idx < out_row_cnt)
    {
        int out_elem_idx = out_row_idx * out_col_cnt + out_col_idx;
        int in_row_idx = out_row_idx;
        int w_col_idx = out_col_idx;

        for (int in_col_idx = 0; in_col_idx < in_col_cnt; in_col_idx++)
        {
            int w_row_idx = in_col_idx;
            out[out_elem_idx] += (in[in_row_idx * in_col_cnt + in_col_idx] * w[w_row_idx * out_col_cnt + w_col_idx]);
        }

        out[out_elem_idx] += b[w_col_idx];
    }
}

__global__ void k_linear_inc_param_derivatives(float *in, float *n, float *w, float *b, float *dw, float *db,
                                               int in_row_cnt, int in_col_cnt, int n_col_cnt, int w_row_cnt, int w_col_cnt)
{
    int w_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int w_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_col_idx < w_col_cnt && w_row_idx < w_row_cnt)
    {
        int w_elem_idx = w_row_idx * w_col_cnt + w_col_idx;
        int n_col_idx = w_row_idx;
        int in_col_idx = w_col_idx;

        for (int i = 0; i < in_row_cnt; i++)
        {
            dw[w_elem_idx] += (in[i * in_col_cnt + in_col_idx] * n[i * n_col_cnt + n_col_idx]);

            if (w_row_idx == 0)
            {
                db[w_col_idx] += in[i * in_col_cnt + in_col_idx];
            }
        }
    }
}

__global__ void k_linear_agg_derivatives(float *in, float *w, float *out, int in_col_cnt, int w_col_cnt, int out_row_cnt, int out_col_cnt)
{
    int out_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_col_idx < out_col_cnt && out_row_idx < out_row_cnt)
    {
        int out_elem_idx = out_row_idx * w_col_cnt + out_col_idx;
        int in_row_idx = out_row_idx;
        int w_row_idx = out_col_idx;

        for (int in_col_idx = 0; in_col_idx < in_col_cnt; in_col_idx++)
        {
            int w_col_idx = in_col_idx;
            out[out_elem_idx] += (in[in_row_idx * in_col_cnt + in_col_idx] * w[w_row_idx * w_col_cnt + w_col_idx]);
        }
    }
}

__device__ float d_sigmoid_evaluate(float val)
{
    return (1.0f / (1.0f + exp(-val)));
}

__device__ float d_sigmoid_derive(float val)
{
    float sigmoid_val = d_sigmoid_evaluate(val);
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

__global__ void k_sigmoid_derive(float *in, float *n, float *out, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        out[tid] = in[tid] * d_sigmoid_derive(n[tid]);
    }
}

Layer::~Layer()
{
    delete this->n_;
}

NdArray *Layer::n()
{
    return this->n_;
}

void Layer::set_n(NdArray *n)
{
    this->n_->copy(n);
}

Learnable::~Learnable()
{
    delete this->w_;
    delete this->b_;
    delete this->dw_;
    delete this->db_;
}

Linear::Linear(int in_cnt, int out_cnt)
{
    this->n_ = new NdArray(true, 1, in_cnt);

    this->w_ = new NdArray(true, out_cnt, in_cnt);
    this->b_ = new NdArray(true, out_cnt);
    this->dw_ = new NdArray(true, out_cnt, in_cnt);
    this->db_ = new NdArray(true, out_cnt);
}

void Linear::forward(NdArray *out)
{
    out->zeros();

    unsigned int grid_row_cnt = (this->n_->rows() / THREADS_PER_BLOCK) + 1;
    unsigned int grid_col_cnt = (out->cols() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_linear_matmul_w_bias<<<grid_dims, block_dims>>>(this->n_->data(), this->w_->data(), out->data(), this->b_->data(),
                                                      this->n_->cols(), this->n_->rows(), out->cols());
}

NdArray *Linear::backward(NdArray *in)
{
    {
        unsigned int grid_row_cnt = (this->w_->rows() / THREADS_PER_BLOCK) + 1;
        unsigned int grid_col_cnt = (this->w_->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), this->w_->data(), this->b_->data(), this->dw_->data(), this->db_->data(),
                                                                  in->rows(), in->cols(), this->n_->cols(), this->w_->rows(), this->w_->cols());
    }

    NdArray *out = new NdArray(true, this->n_->rows(), this->n_->cols());
    out->zeros();

    {
        unsigned int grid_row_cnt = (out->rows() / THREADS_PER_BLOCK) + 1;
        unsigned int grid_col_cnt = (out->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), this->w_->data(), out->data(),
                                                            in->cols(), this->w_->cols(), out->rows(), out->cols());
    }

    delete in;
    return out;
}

Sigmoid::Sigmoid(int in_cnt)
{
    this->n_ = new NdArray(true, 1, in_cnt);
}

void Sigmoid::forward(NdArray *out)
{
    out->zeros();
    k_sigmoid_evaluate<<<this->n_->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(this->n_->data(), out->data(), this->n_->count());
}

NdArray *Sigmoid::backward(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->rows(), this->n_->cols());

    k_sigmoid_derive<<<in->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(in->data(), this->n_->data(), out->data(), in->count());

    delete in;
    return out;
}