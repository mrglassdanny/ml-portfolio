#include "layer.cuh"

// Device functions:

// Kernel functions:

__global__ void k_linear_matmul_w_bias(float *in, float *w, float *out, float *b,
                                       int in_col_cnt, int out_col_cnt, int out_elem_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int out_elem_idx = tid;

    if (out_elem_idx < out_elem_cnt)
    {
        int in_row_idx = out_elem_idx / out_col_cnt;
        int w_col_idx = out_elem_idx % out_col_cnt;

        for (int in_col_idx = 0; in_col_idx < in_col_cnt; in_col_idx++)
        {
            out[out_elem_idx] += (in[in_row_idx * in_col_cnt + in_col_idx] * w[w_col_idx + (in_col_idx * out_col_cnt)]);
        }

        out[out_elem_idx] += b[w_col_idx];
    }
}

LinearLayer::LinearLayer(int in_cnt, int out_cnt)
{
    this->n_ = new Array2d(false, in_cnt, 5);
    this->w_ = new Array2d(false, in_cnt, out_cnt);
    this->b_ = new Array1d(false, out_cnt);

    this->w_->rands(0.0f, sqrt(1.0f / in_cnt));
    this->b_->zeros();
}

LinearLayer::~LinearLayer()
{
    delete this->n_;
    delete this->w_;
    delete this->b_;
}

void LinearLayer::forward(Array2d *out)
{
    out->zeros();

    int batch_size = this->n_->rows();
    int in_cnt = this->n_->cols();
    int out_cnt = out->cols();

    {
        int num_blocks = ((batch_size * out_cnt) / THREADS_PER_BLOCK) + 1;
        k_linear_matmul_w_bias<<<num_blocks, THREADS_PER_BLOCK>>>(this->n_->data(), this->w_->data(), out->data(), this->b_->data(), in_cnt, out_cnt, (batch_size * out_cnt));
    }
}

Array2d *LinearLayer::backward(Array2d *d_l)
{
    return NULL;
}

ActivationLayer::ActivationLayer(int in_cnt)
{
}

ActivationLayer::~ActivationLayer()
{
}
