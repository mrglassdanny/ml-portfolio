#include "layer.cuh"

// Device functions:

// Kernel functions:

__global__ void k_linear_matmul_w_bias(float *in_mtx, float *w_mtx, float *out_mtx, float *b_vec,
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
            out_mtx[out_elem_idx] += (in_mtx[in_row_idx * in_col_cnt + in_col_idx] * w_mtx[w_col_idx + (in_col_idx * out_col_cnt)]);
        }

        out_mtx[out_elem_idx] += b_vec[w_col_idx];
    }
}

// Layer:

Layer::Layer()
{
    this->n = NULL;
}

Layer::~Layer()
{
    delete this->n;
}

LinearLayer::LinearLayer(int in_cnt, int out_cnt)
{
    this->w = new Tensor(false, Dimensions(in_cnt, out_cnt));
    this->b = new Tensor(false, Dimensions(out_cnt));

    this->w->rands(0.0f, sqrt(1.0f / in_cnt));
    this->b->zeros();
}

LinearLayer::~LinearLayer()
{
    delete this->w;
    delete this->b;
}

void LinearLayer::forward(Tensor *out)
{
    out->zeros();

    int batch_size = this->n->get_dims().get_dim(0);
    int in_cnt = this->n->get_dims().get_dim(1);
    int out_cnt = out->get_dims().get_dim(1);

    {
        int num_blocks = ((batch_size * out_cnt) / THREADS_PER_BLOCK) + 1;
        k_linear_matmul_w_bias<<<num_blocks, THREADS_PER_BLOCK>>>(this->n->get_data(), this->w->get_data(), out->get_data(), this->b->get_data(), in_cnt, out_cnt, (batch_size * out_cnt));
    }
}

Tensor *LinearLayer::backward(Tensor *d_l)
{
    return NULL;
}