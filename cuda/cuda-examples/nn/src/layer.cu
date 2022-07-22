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

__global__ void k_activation_sigmoid_evaluate(float *in, float *out, int in_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < in_cnt)
    {
        out[tid] = d_sigmoid(in[tid]);
    }
}

__global__ void k_activation_sigmoid_derive(float *in, float *out, int in_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < in_cnt)
    {
        out[tid] = d_derive_sigmoid(in[tid]);
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

void SigmoidActivation::evaluate(Array2d *in, Array2d *out)
{
    {
        int num_blocks = (in->count() / THREADS_PER_BLOCK) + 1;
        k_activation_sigmoid_evaluate<<<num_blocks, THREADS_PER_BLOCK>>>(in->data(), out->data(), in->count());
    }
}

void SigmoidActivation::derive(Array2d *in, Array2d *out)
{
    {
        int num_blocks = (in->count() / THREADS_PER_BLOCK) + 1;
        k_activation_sigmoid_derive<<<num_blocks, THREADS_PER_BLOCK>>>(in->data(), out->data(), in->count());
    }
}

ActivationLayer::ActivationLayer(Activation *a, int in_cnt)
{
    this->n_ = new Array2d(false, 1, in_cnt);
    this->a_ = a;
}

ActivationLayer::~ActivationLayer()
{
}

void ActivationLayer::forward(Array2d *out)
{
    this->a_->evaluate(this->n_, out);
}

Array2d *ActivationLayer::backward(Array2d *d_l)
{
    return NULL;
}