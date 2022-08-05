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

__global__ void k_conv2d_pad()
{
}

__device__ void d_conv2d_convolve(float *in, float *w, float *out, int channel_cnt, int in_row_cnt, int in_col_cnt, int in_cnt,
                                  int filter_row_cnt, int filter_col_cnt, int w_cnt, int out_row_cnt, int out_col_cnt,
                                  int stride_row_cnt, int stride_col_cnt)
{
    for (int i = 0; i < channel_cnt; i++)
    {
        for (int j = 0; j < out_row_cnt; j++)
        {
            for (int k = 0; k < out_col_cnt; k++)
            {
                int out_elem_idx = j * out_col_cnt + k;
                int in_row_offset = j + stride_row_cnt;
                int in_col_offset = k + stride_col_cnt;

                for (int l = 0; l < filter_row_cnt; l++)
                {
                    for (int m = 0; m < filter_col_cnt; m++)
                    {
                        int in_elem_idx = (i * in_cnt) + (l + in_row_offset) * filter_col_cnt + (m + in_col_offset);
                        int w_elem_idx = (i * w_cnt) + (l * filter_col_cnt + m);
                        out[out_elem_idx] += (in[in_elem_idx] * w[w_elem_idx]);
                    }
                }
            }
        }
    }
}

__global__ void k_conv2d_evaluate(float *in, float *w, float *out, int batch_size, int channel_cnt, int in_row_cnt, int in_col_cnt,
                                  int filter_cnt, int filter_row_cnt, int filter_col_cnt, int out_row_cnt, int out_col_cnt,
                                  int stride_row_cnt, int stride_col_cnt)
{
    int filter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < batch_size && filter_idx < filter_cnt)
    {
        int in_cnt = in_row_cnt * in_col_cnt;
        int w_cnt = filter_row_cnt * filter_col_cnt;
        int out_cnt = out_row_cnt * out_col_cnt;

        float *l_in = &in[(batch_idx * channel_cnt * in_cnt)];
        float *l_w = &w[(filter_idx * channel_cnt * w_cnt)];
        float *l_out = &out[(batch_idx * filter_cnt * out_cnt)];

        d_conv2d_convolve(l_in, l_w, l_out, channel_cnt, in_row_cnt, in_col_cnt, in_cnt,
                          filter_row_cnt, filter_col_cnt, w_cnt, out_row_cnt, out_col_cnt,
                          stride_row_cnt, stride_col_cnt);
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

__global__ void k_sigmoid_evaluate(float *in, float *out, int row_cnt, int col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        int elem_idx = row_idx * col_cnt + col_idx;

        out[elem_idx] = d_sigmoid_evaluate(in[elem_idx]);
    }
}

__global__ void k_sigmoid_derive(float *in, float *n, float *out, int row_cnt, int col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        int elem_idx = row_idx * col_cnt + col_idx;

        out[elem_idx] = in[elem_idx] * d_sigmoid_derive(n[elem_idx]);
    }
}

Layer::~Layer()
{
    delete this->n_;
}

int Layer::batch_size()
{
    return this->n_->shape()[0];
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
    this->n_ = new NdArray(true, Shape(DEFAULT_BATCH_SIZE, in_cnt));
    this->params_ = new Parameters(Shape(in_cnt, out_cnt), Shape(out_cnt), in_cnt, out_cnt);
}

void Linear::evaluate(NdArray *out)
{
    out->zeros();

    int grid_row_cnt = (out->shape()[0] / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (out->shape()[1] / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();

    k_linear_matmul_w_bias<<<grid_dims, block_dims>>>(n->data(), w->data(), out->data(), b->data(),
                                                      n->shape()[1], out->shape()[0], out->shape()[1]);
}

NdArray *Linear::derive(NdArray *in)
{
    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();
    NdArray *dw = this->params_->weight_gradients();
    NdArray *db = this->params_->bias_gradients();

    {
        int grid_row_cnt = (w->shape()[0] / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (w->shape()[1] / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), n->data(), w->data(), b->data(), dw->data(), db->data(),
                                                                  in->shape()[0], in->shape()[1], n->shape()[1], w->shape()[0], w->shape()[1]);
    }

    NdArray *out = NdArray::zeros(true, n->shape());

    {
        int grid_row_cnt = (out->shape()[0] / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (out->shape()[1] / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), out->data(),
                                                            in->shape()[1], w->shape()[1], out->shape()[0], out->shape()[1]);
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

Conv2d::Conv2d(int channel_cnt, Shape in_shape, int filter_cnt, Shape filter_shape, Shape padding_shape, Shape stride_shape)
{
    this->n_ = new NdArray(true, Shape(DEFAULT_BATCH_SIZE, in_shape));

    int out_row_cnt = (((in_shape[0] - filter_shape[0]) + (2 * padding_shape[0])) / stride_shape[0]) + 1;
    int out_col_cnt = (((in_shape[1] - filter_shape[1]) + (2 * padding_shape[1])) / stride_shape[1]) + 1;

    this->params_ = new Parameters(Shape(filter_cnt, channel_cnt, filter_shape[0], filter_shape[1]), Shape(filter_cnt, channel_cnt), in_shape[0], in_shape[1]);

    this->channel_cnt_ = channel_cnt;
    this->in_shape_ = in_shape;
    this->filter_cnt_ = filter_cnt;
    this->filter_shape_ = filter_shape;
    this->padding_shape_ = padding_shape;
    this->stride_shape_ = stride_shape;
}

void Conv2d::evaluate(NdArray *out)
{
    out->zeros();
    // this->lock_padding();

    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->filter_cnt_ / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();

    int out_row_cnt = (((this->in_shape_[0] - this->filter_shape_[0]) + (2 * this->padding_shape_[0])) / this->stride_shape_[0]) + 1;
    int out_col_cnt = (((this->in_shape_[1] - this->filter_shape_[1]) + (2 * this->padding_shape_[1])) / this->stride_shape_[1]) + 1;

    k_conv2d_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), out->data(), this->batch_size(), this->channel_cnt_, this->in_shape_[0], this->in_shape_[1],
                                                 this->filter_cnt_, this->filter_shape_[0], this->filter_shape_[1], out_row_cnt, out_col_cnt,
                                                 this->stride_shape_[0], this->stride_shape_[1]);
}

NdArray *Conv2d::derive(NdArray *in)
{
    return nullptr;
}

Shape Conv2d::input_shape()
{
    return Shape(this->batch_size(), this->channel_cnt_, this->in_shape_[0], this->in_shape_[1]);
}

Shape Conv2d::output_shape()
{
    int out_row_cnt = (((this->in_shape_[0] - this->filter_shape_[0]) + (2 * this->padding_shape_[0])) / this->stride_shape_[0]) + 1;
    int out_col_cnt = (((this->in_shape_[1] - this->filter_shape_[1]) + (2 * this->padding_shape_[1])) / this->stride_shape_[1]) + 1;

    return Shape(this->batch_size(), this->filter_cnt_, out_row_cnt, out_col_cnt);
}

Activation::Activation(int in_cnt)
{
    this->n_ = new NdArray(true, Shape(DEFAULT_BATCH_SIZE, in_cnt));
}

Shape Activation::input_shape()
{
    return this->n_->shape();
}

Shape Activation::output_shape()
{
    return this->n_->shape();
}

Sigmoid::Sigmoid(int in_cnt)
    : Activation(in_cnt)
{
}

void Sigmoid::evaluate(NdArray *out)
{
    out->zeros();

    int grid_row_cnt = (out->shape()[0] / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (out->shape()[1] / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_sigmoid_evaluate<<<grid_dims, block_dims>>>(this->n_->data(), out->data(), out->shape()[0], out->shape()[1]);
}

NdArray *Sigmoid::derive(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->shape());

    int grid_row_cnt = (out->shape()[0] / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (out->shape()[1] / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_sigmoid_derive<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), out->data(), out->shape()[0], out->shape()[1]);

    delete in;
    return out;
}
