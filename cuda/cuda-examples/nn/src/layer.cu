#include "layer.cuh"

using namespace nn::layer;

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

__global__ void k_linear_inc_param_derivatives(float *in, float *n, float *dw, float *db,
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

__global__ void k_linear_agg_derivatives(float *in, float *w, float *out, int batch_size, int in_cnt, int w_col_cnt, int out_cnt)
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

__global__ void k_conv2d_evaluate(float *in, float *w, float *b, float *out, int batch_size, int channel_cnt, int in_row_cnt, int in_col_cnt,
                                  int filter_cnt, int filter_row_cnt, int filter_col_cnt, int out_row_cnt, int out_col_cnt,
                                  int stride_row_cnt, int stride_col_cnt)
{
    int filter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (filter_idx < filter_cnt && batch_idx < batch_size)
    {
        int in_cnt = in_row_cnt * in_col_cnt;
        int w_cnt = filter_row_cnt * filter_col_cnt;
        int out_cnt = out_row_cnt * out_col_cnt;

        float *l_in = &in[(batch_idx * channel_cnt * in_cnt)];
        float *l_w = &w[(filter_idx * channel_cnt * w_cnt)];
        float *l_b = &b[(filter_idx * channel_cnt)];
        float *l_out = &out[((batch_idx * filter_cnt * out_cnt) + (filter_idx * out_cnt))];

        for (int out_row_idx = 0; out_row_idx < out_row_cnt; out_row_idx++)
        {
            for (int out_col_idx = 0; out_col_idx < out_col_cnt; out_col_idx++)
            {
                for (int channel_idx = 0; channel_idx < channel_cnt; channel_idx++)
                {
                    for (int w_row_idx = 0; w_row_idx < filter_row_cnt; w_row_idx++)
                    {
                        for (int w_col_idx = 0; w_col_idx < filter_col_cnt; w_col_idx++)
                        {
                            l_out[out_row_idx * out_col_cnt + out_col_idx] +=
                                (l_in[(channel_idx * in_cnt) + (((w_row_idx + (out_row_idx * stride_row_cnt)) * in_col_cnt + (w_col_idx + (out_col_idx * stride_col_cnt))))] *
                                 l_w[(channel_idx * w_cnt) + (w_row_idx * filter_col_cnt + w_col_idx)]);
                        }
                    }

                    l_out[out_row_idx * out_col_cnt + out_col_idx] += l_b[channel_idx];
                }
            }
        }
    }
}

__global__ void k_conv2d_inc_param_derivatives(float *in, float *n, float *dw, float *db, int batch_size, int channel_cnt, int filter_cnt,
                                               int in_row_cnt, int in_col_cnt, int in_cnt, int n_row_cnt, int n_col_cnt, int n_cnt, int w_row_cnt, int w_col_cnt,
                                               int w_cnt, int stride_row_cnt, int stride_col_cnt)
{
    int channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int filter_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (channel_idx < channel_cnt && filter_idx < filter_cnt)
    {
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            for (int in_row_idx = 0; in_row_idx < in_row_cnt; in_row_idx++)
            {
                for (int in_col_idx = 0; in_col_idx < in_col_cnt; in_col_idx++)
                {
                    int in_elem_idx = (batch_idx * filter_cnt * in_cnt) + (filter_idx * in_cnt) + (in_row_idx * in_col_cnt + in_col_idx);

                    for (int w_row_idx = 0; w_row_idx < w_row_cnt; w_row_idx++)
                    {
                        for (int w_col_idx = 0; w_col_idx < w_col_cnt; w_col_idx++)
                        {
                            int w_elem_idx = (filter_idx * channel_cnt * w_cnt) + (channel_idx * w_cnt) + (w_row_idx * w_col_cnt + w_col_idx);

                            dw[w_elem_idx] +=
                                (in[in_elem_idx] *
                                 n[(batch_idx * channel_cnt * n_cnt) + (channel_idx * n_cnt) + ((w_row_idx + (in_row_idx * stride_row_cnt)) * n_col_cnt + (w_col_idx + (in_col_idx * stride_col_cnt)))]);
                        }
                    }

                    db[filter_idx * channel_cnt + channel_idx] += in[in_elem_idx];
                }
            }
        }
    }
}

__global__ void k_conv2d_agg_derivatives(float *in, float *w, float *out, int batch_size, int channel_cnt, int filter_cnt,
                                         int in_row_cnt, int in_col_cnt, int in_cnt, int w_row_cnt, int w_col_cnt, int w_cnt, int out_row_cnt, int out_col_cnt, int out_cnt,
                                         int stride_row_cnt, int stride_col_cnt)
{
    int channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (channel_idx < channel_cnt && batch_idx < batch_size)
    {
        for (int filter_idx = 0; filter_idx < filter_cnt; filter_idx++)
        {
            for (int in_row_idx = 0; in_row_idx < in_row_cnt; in_row_idx++)
            {
                for (int in_col_idx = 0; in_col_idx < in_col_cnt; in_col_idx++)
                {
                    int in_elem_idx = (batch_idx * filter_cnt * in_cnt) + (filter_idx * in_cnt) + (in_row_idx * in_col_cnt + in_col_idx);

                    for (int w_row_idx = 0; w_row_idx < w_row_cnt; w_row_idx++)
                    {
                        for (int w_col_idx = 0; w_col_idx < w_col_cnt; w_col_idx++)
                        {
                            int out_elem_idx = (batch_idx * channel_cnt * out_cnt) + (channel_idx * out_cnt) + ((w_row_idx + (in_row_idx * stride_row_cnt)) * out_col_cnt + (w_col_idx + (in_col_idx * stride_col_cnt)));

                            out[out_elem_idx] +=
                                (in[in_elem_idx] *
                                 w[(filter_idx * channel_cnt * w_cnt) + (channel_idx * w_cnt) + (w_row_idx * w_col_cnt + w_col_idx)]);
                        }
                    }
                }
            }
        }
    }
}

__device__ float d_sigmoid_evaluate(float val)
{
    return (1.0f / (1.0f + exp(-val)));
}

__device__ float d_sigmoid_derive(float sigmoid_val)
{
    return (sigmoid_val) * (1.0f - sigmoid_val);
}

__global__ void k_sigmoid_evaluate(float *in, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = d_sigmoid_evaluate(in[elem_idx]);
    }
}

__global__ void k_sigmoid_derive(float *in, float *n, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = in[elem_idx] * d_sigmoid_derive(d_sigmoid_evaluate(n[elem_idx]));
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

__global__ void k_tanh_evaluate(float *in, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = d_tanh_evaluate(in[elem_idx]);
    }
}

__global__ void k_tanh_derive(float *in, float *n, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = in[elem_idx] * d_tanh_derive(d_tanh_evaluate(n[elem_idx]));
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

__global__ void k_relu_evaluate(float *in, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = d_relu_evaluate(in[elem_idx]);
    }
}

__global__ void k_relu_derive(float *in, float *n, float *out, int batch_size, int cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cnt && batch_idx < batch_size)
    {
        int elem_idx = batch_idx * cnt + idx;

        out[elem_idx] = in[elem_idx] * d_relu_derive(d_relu_evaluate(n[elem_idx]));
    }
}

Layer::~Layer()
{
    delete this->n_;
}

void Layer::summarize()
{
    std::string cls_name(typeid(*this).name());
    for (int i = cls_name.size(); i < 26; i++)
    {
        cls_name.push_back(' ');
    }

    printf("%s\t", cls_name.c_str());
    this->input_shape().print();
    printf(" -> ");
    this->output_shape().print();
}

int Layer::batch_size()
{
    return this->n_->shape()[0];
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

size_t Parameters::count()
{
    return this->w_->count() + this->b_->count();
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

Linear::Linear(Shape in_shape, Shape out_shape)
{
    this->n_ = new NdArray(true, in_shape);

    int in_cnt = (in_shape.dims_size() / this->batch_size());
    int out_cnt = (out_shape.dims_size() / this->batch_size());

    this->params_ = new Parameters(Shape(in_cnt, out_cnt), Shape(out_cnt), in_cnt, out_cnt);
}

void Linear::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->out_features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();

    k_linear_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), b->data(), out->data(),
                                                 this->batch_size(), this->in_features(), this->out_features());
}

NdArray *Linear::derive(NdArray *in)
{
    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();
    NdArray *dw = this->params_->weight_gradients();
    NdArray *db = this->params_->bias_gradients();

    {
        int grid_row_cnt = (this->weight_rows() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->weight_cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), n->data(), dw->data(), db->data(),
                                                                  this->batch_size(), this->out_features(), this->in_features(),
                                                                  this->weight_rows(), this->weight_cols());
    }

    NdArray *out = NdArray::zeros(true, this->input_shape());

    {
        int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->in_features() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_linear_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), out->data(),
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

Conv2d::Conv2d(Shape in_shape, Shape filter_shape, Padding padding, Stride stride)
{
    this->n_ = new NdArray(true, in_shape);
    this->params_ = new Parameters(filter_shape, Shape(filter_shape[0], filter_shape[1]), this->in_feature_rows(), this->in_feature_cols());
    this->padding_ = padding;
    this->stride_ = stride;
}

void Conv2d::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->filters() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();

    k_conv2d_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), b->data(), out->data(), this->batch_size(), this->channels(), this->in_feature_rows(), this->in_feature_cols(),
                                                 this->filters(), this->filter_rows(), this->filter_cols(), this->out_feature_rows(), this->out_feature_cols(),
                                                 this->stride_rows(), this->stride_cols());
}

NdArray *Conv2d::derive(NdArray *in)
{
    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();
    NdArray *dw = this->params_->weight_gradients();
    NdArray *db = this->params_->bias_gradients();

    {
        int grid_row_cnt = (this->filters() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->channels() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_conv2d_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), n->data(), dw->data(), db->data(), this->batch_size(), this->channels(), this->filters(),
                                                                  this->out_feature_rows(), this->out_feature_cols(), (this->out_feature_rows() * this->out_feature_cols()),
                                                                  this->in_feature_rows(), this->in_feature_cols(), (this->in_feature_rows() * this->in_feature_cols()),
                                                                  this->filter_rows(), this->filter_cols(), (this->filter_rows() * this->filter_cols()),
                                                                  this->stride_rows(), this->stride_cols());
    }

    NdArray *out = NdArray::zeros(true, this->input_shape());

    {
        int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->channels() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_conv2d_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), out->data(), this->batch_size(), this->channels(), this->filters(),
                                                            this->out_feature_rows(), this->out_feature_cols(), (this->out_feature_rows() * this->out_feature_cols()),
                                                            this->filter_rows(), this->filter_cols(), (this->filter_rows() * this->filter_cols()),
                                                            this->in_feature_rows(), this->in_feature_cols(), (this->in_feature_rows() * this->in_feature_cols()),
                                                            this->stride_rows(), this->stride_cols());
    }

    delete in;
    return out;
}

Shape Conv2d::input_shape()
{
    return this->n_->shape();
}

Shape Conv2d::output_shape()
{
    int out_row_cnt = (((this->in_feature_rows() - this->filter_rows()) + (2 * this->padding_rows())) / this->stride_rows()) + 1;
    int out_col_cnt = (((this->in_feature_cols() - this->filter_cols()) + (2 * this->padding_cols())) / this->stride_cols()) + 1;

    return Shape(this->batch_size(), this->filters(), out_row_cnt, out_col_cnt);
}

void Conv2d::validate()
{

    int r = this->filter_rows();
    while (r < this->in_feature_rows())
    {
        r += this->stride_rows();
    }

    if (r != this->in_feature_rows())
    {
        printf("CONV2D LAYER VALIDATION FAILED: filter/stride row combination does not fit input row count\n");
        exit(EXIT_FAILURE);
    }

    int c = this->filter_cols();
    while (c < this->in_feature_cols())
    {
        c += this->stride_cols();
    }

    if (c != this->in_feature_cols())
    {
        printf("CONV2D LAYER VALIDATION FAILED: filter/stride column combination does not fit input column count\n");
        exit(EXIT_FAILURE);
    }

    if (this->filter_rows() < this->stride_rows())
    {
        printf("CONV2D LAYER VALIDATION FAILED: filter row count less than stride row count\n");
        exit(EXIT_FAILURE);
    }

    if (this->filter_cols() < this->stride_cols())
    {
        printf("CONV2D LAYER VALIDATION FAILED: filter column count less than stride column count\n");
        exit(EXIT_FAILURE);
    }
}

int Conv2d::channels()
{
    return this->n_->shape()[1];
}

int Conv2d::in_feature_rows()
{
    return this->n_->shape()[2];
}

int Conv2d::in_feature_cols()
{
    return this->n_->shape()[3];
}

int Conv2d::filters()
{
    return this->params_->weights()->shape()[0];
}

int Conv2d::filter_rows()
{
    return this->params_->weights()->shape()[2];
}

int Conv2d::filter_cols()
{
    return this->params_->weights()->shape()[3];
}

int Conv2d::padding_rows()
{
    return this->padding_.row_cnt;
}

int Conv2d::padding_cols()
{
    return this->padding_.col_cnt;
}

int Conv2d::stride_rows()
{
    return this->stride_.row_cnt;
}

int Conv2d::stride_cols()
{
    return this->stride_.col_cnt;
}

int Conv2d::out_feature_rows()
{
    return this->output_shape()[2];
}

int Conv2d::out_feature_cols()
{
    return this->output_shape()[3];
}

Activation::Activation(Shape shape)
{
    this->n_ = new NdArray(true, shape);
}

Shape Activation::input_shape()
{
    return this->n_->shape();
}

Shape Activation::output_shape()
{
    return this->n_->shape();
}

void Activation::validate() {}

int Activation::features()
{
    return (this->n_->shape().dims_size() / this->batch_size());
}

Sigmoid::Sigmoid(Shape shape)
    : Activation(shape)
{
}

void Sigmoid::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_sigmoid_evaluate<<<grid_dims, block_dims>>>(this->n_->data(), out->data(), this->batch_size(), this->features());
}

NdArray *Sigmoid::derive(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->shape());

    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_sigmoid_derive<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), out->data(), this->batch_size(), this->features());

    delete in;
    return out;
}

Tanh::Tanh(Shape shape)
    : Activation(shape)
{
}

void Tanh::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_tanh_evaluate<<<grid_dims, block_dims>>>(this->n_->data(), out->data(), this->batch_size(), this->features());
}

NdArray *Tanh::derive(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->shape());

    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_tanh_derive<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), out->data(), this->batch_size(), this->features());

    delete in;
    return out;
}

ReLU::ReLU(Shape shape)
    : Activation(shape)
{
}

void ReLU::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_relu_evaluate<<<grid_dims, block_dims>>>(this->n_->data(), out->data(), this->batch_size(), this->features());
}

NdArray *ReLU::derive(NdArray *in)
{
    NdArray *out = new NdArray(true, this->n_->shape());

    int grid_row_cnt = (this->batch_size() / THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->features() / THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    k_relu_derive<<<grid_dims, block_dims>>>(in->data(), this->n_->data(), out->data(), this->batch_size(), this->features());

    delete in;
    return out;
}
