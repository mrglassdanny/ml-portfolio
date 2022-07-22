#include "layer.cuh"

// Device functions:

__device__ float d_sigmoid(float val)
{
    return (1.0f / (1.0f + exp(-val)));
}

__device__ float d_derive_sigmoid(float sigmoid_val)
{
    return (sigmoid_val) * (1.0f - sigmoid_val);
}

// Kernel functions:

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

__global__ void k_activation_sigmoid_evaluate(float *in, float *out, int in_row_cnt, int in_col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < in_col_cnt && row_idx < in_row_cnt)
    {
        int out_elem_idx = row_idx * in_col_cnt + col_idx;

        out[out_elem_idx] = d_sigmoid(in[out_elem_idx]);
    }
}

__global__ void k_activation_sigmoid_derive(float *in, float *out, int in_row_cnt, int in_col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < in_col_cnt && row_idx < in_row_cnt)
    {
        int out_elem_idx = row_idx * in_col_cnt + col_idx;

        out[out_elem_idx] = d_derive_sigmoid(in[out_elem_idx]);
    }
}

LinearLayer::LinearLayer(int in_cnt, int out_cnt)
{
    this->n_ = new Array2d(true, 1, in_cnt);
    this->w_ = new Array2d(true, in_cnt, out_cnt);
    this->b_ = new Array1d(true, out_cnt);
    this->dw_ = new Array2d(true, in_cnt, out_cnt);
    this->db_ = new Array1d(true, out_cnt);

    this->w_->rands(0.0f, sqrt(1.0f / in_cnt));
    this->b_->zeros();
    this->dw_->zeros();
    this->db_->zeros();
}

LinearLayer::~LinearLayer()
{
    delete this->n_;
    delete this->w_;
    delete this->b_;
    delete this->dw_;
    delete this->db_;
}

void LinearLayer::forward(Array2d *out)
{
    out->zeros();

    {
        unsigned int grid_row_cnt = (this->n_->rows() / THREADS_PER_BLOCK) + 1;
        unsigned int grid_col_cnt = (out->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_matmul_w_bias<<<grid_dims, block_dims>>>(this->n_->data(), this->w_->data(), out->data(), this->b_->data(),
                                                          this->n_->cols(), this->n_->rows(), out->cols());
    }
}

Array2d *LinearLayer::backward(Array2d *in)
{
    {
        unsigned int grid_row_cnt = (this->w_->rows() / THREADS_PER_BLOCK) + 1;
        unsigned int grid_col_cnt = (this->w_->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), this->w_->data(), this->b_->data(), this->dw_->data(), this->db_->data(),
                                                                  in->rows(), in->cols(), this->n_->cols(), this->w_->rows(), this->w_->cols());
    }

    Array2d *out = new Array2d(true, this->n_->rows(), this->n_->cols());
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

Array2d *LinearLayer::n()
{
    return this->n_;
}

void SigmoidActivator::evaluate(Array2d *in, Array2d *out)
{
    {
        unsigned int grid_row_cnt = (in->rows() / THREADS_PER_BLOCK) + 1;
        unsigned int grid_col_cnt = (in->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_activation_sigmoid_evaluate<<<grid_dims, block_dims>>>(in->data(), out->data(), in->rows(), in->cols());
    }
}

void SigmoidActivator::derive(Array2d *in, Array2d *out)
{
    {
        unsigned int grid_row_cnt = (in->rows() / THREADS_PER_BLOCK) + 1;
        unsigned int grid_col_cnt = (in->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_activation_sigmoid_derive<<<grid_dims, block_dims>>>(in->data(), out->data(), in->rows(), in->cols());
    }
}

ActivationLayer::ActivationLayer(Activator *a, int in_cnt)
{
    this->n_ = new Array2d(true, 1, in_cnt);
    this->a_ = a;
}

ActivationLayer::~ActivationLayer()
{
    delete this->n_;
    delete this->a_;
}

void ActivationLayer::forward(Array2d *out)
{
    out->zeros();

    this->a_->evaluate(this->n_, out);
}

Array2d *ActivationLayer::backward(Array2d *in)
{
    Array2d *out = new Array2d(true, in->rows(), in->cols());

    this->a_->derive(in, out);

    delete in;
    return out;
}

Array2d *ActivationLayer::n()
{
    return this->n_;
}