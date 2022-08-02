#include "layer.cuh"

#define DEFAULT_BATCH_SIZE 1

using namespace nn::layer;

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
                int b_elem_idx = w_col_idx;
                db[b_elem_idx] += in[i * in_col_cnt + in_col_idx];
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

Parameters::Parameters(Shape w_shape, Shape b_shape, int fan_in, int fan_out)
{
    this->w_ = NdArray::rands(true, w_shape, 0.0f, sqrt(1.0f / fan_in));
    this->b_ = NdArray::zeros(true, b_shape);
    this->dw_ = NdArray::zeros(true, w_shape);
    this->db_ = NdArray::zeros(true, b_shape);
}

Parameters::~Parameters()
{
    delete this->w_;
    delete this->b_;
    delete this->dw_;
    delete this->db_;
}

void Parameters::zero_grad()
{
    this->dw_->zeros();
    this->db_->zeros();
}

NdArray *Parameters::weights()
{
    return this->w_;
}

NdArray *Parameters::biases()
{
    return this->b_;
}

NdArray *Parameters::weight_gradients()
{
    return this->dw_;
}

NdArray *Parameters::bias_gradients()
{
    return this->db_;
}

Layer::~Layer()
{
    delete this->n_;
}

int Layer::batch_size()
{
    return this->n_->shape().dim(0);
}

void Layer::lock_batch_size(int batch_size)
{
    this->n_->change_dim(0, batch_size);
}

NdArray *Layer::neurons()
{
    return this->n_;
}

void Layer::copy_neurons(NdArray *n)
{
    this->n_->copy(n);
}

Learnable::~Learnable()
{
    delete params_;
}

Parameters *Learnable::parameters()
{
    return this->params_;
}

Linear::Linear(int in_cnt, int out_cnt)
{
    this->n_ = new NdArray(true, DEFAULT_BATCH_SIZE, in_cnt);
    this->params_ = new Parameters(Shape(in_cnt, out_cnt), Shape(out_cnt), in_cnt, out_cnt);
}

void Linear::evaluate(NdArray *out)
{
    out->zeros();

    int grid_row_cnt = (this->n_->rows() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (out->cols() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();
    NdArray *dw = this->params_->weight_gradients();
    NdArray *db = this->params_->bias_gradients();

    k_linear_matmul_w_bias<<<grid_dims, block_dims>>>(n->data(), w->data(), out->data(), b->data(),
                                                      n->cols(), n->rows(), out->cols());
}

NdArray *Linear::derive(NdArray *in)
{
    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();
    NdArray *dw = this->params_->weight_gradients();
    NdArray *db = this->params_->bias_gradients();

    {
        int grid_row_cnt = (w->rows() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (w->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), n->data(), w->data(), b->data(), dw->data(), db->data(),
                                                                  in->rows(), in->cols(), n->cols(), w->rows(), w->cols());
    }

    NdArray *out = new NdArray(true, this->n_->rows(), this->n_->cols());
    out->zeros();

    {
        int grid_row_cnt = (out->rows() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (out->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), out->data(),
                                                            in->cols(), w->cols(), out->rows(), out->cols());
    }

    delete in;
    return out;
}

Activation::Activation(int in_cnt)
{
    this->n_ = new NdArray(true, DEFAULT_BATCH_SIZE, in_cnt);
}

Sigmoid::Sigmoid(int in_cnt)
    : Activation(in_cnt)
{
}

void Sigmoid::evaluate(NdArray *out)
{
    out->zeros();
    k_sigmoid_evaluate<<<this->n_->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(this->n_->data(), out->data(), this->n_->count());
}

NdArray *Sigmoid::derive(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->rows(), this->n_->cols());

    k_sigmoid_derive<<<in->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(in->data(), this->n_->data(), out->data(), in->count());

    delete in;
    return out;
}