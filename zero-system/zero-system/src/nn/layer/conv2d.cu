#include "conv2d.cuh"

using namespace nn::layer;

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

Conv2d::Conv2d(Shape in_shape, Shape filter_shape, Padding padding, Stride stride)
{
    this->n_ = new NdArray(true, in_shape);
    this->default_n_shape_ = in_shape;
    this->params_ = new Parameters(filter_shape, Shape(filter_shape[0], filter_shape[1]), this->in_feature_rows(), this->in_feature_cols());
    this->padding_ = padding;
    this->stride_ = stride;
}

void Conv2d::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->filters() / CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

    if (this->padding_rows() > 0 || this->padding_cols() > 0)
    {
        NdArray *padded_n = NdArray::pad(this->n_, this->padding_rows(), this->padding_cols());
        delete this->n_;
        this->n_ = padded_n;
    }

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
        int grid_row_cnt = (this->filters() / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->channels() / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_conv2d_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), n->data(), dw->data(), db->data(), this->batch_size(), this->channels(), this->filters(),
                                                                  this->out_feature_rows(), this->out_feature_cols(), (this->out_feature_rows() * this->out_feature_cols()),
                                                                  this->in_feature_rows(), this->in_feature_cols(), (this->in_feature_rows() * this->in_feature_cols()),
                                                                  this->filter_rows(), this->filter_cols(), (this->filter_rows() * this->filter_cols()),
                                                                  this->stride_rows(), this->stride_cols());
    }

    NdArray *out = NdArray::zeros(true, this->input_shape());

    {
        int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->channels() / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_conv2d_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), out->data(), this->batch_size(), this->channels(), this->filters(),
                                                            this->out_feature_rows(), this->out_feature_cols(), (this->out_feature_rows() * this->out_feature_cols()),
                                                            this->filter_rows(), this->filter_cols(), (this->filter_rows() * this->filter_cols()),
                                                            this->in_feature_rows(), this->in_feature_cols(), (this->in_feature_rows() * this->in_feature_cols()),
                                                            this->stride_rows(), this->stride_cols());
    }

    if (this->padding_rows() > 0 || this->padding_cols() > 0)
    {
        NdArray *unpadded_out = NdArray::unpad(out, this->padding_rows(), this->padding_cols());
        delete out;
        out = unpadded_out;
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
    if (this->n_->num_dims() != 4)
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: invalid input shape");
    }

    if (this->filter_shape().num_dims() != 4)
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: invalid filter shape");
    }

    if (this->n_->shape()[1] != this->filter_shape()[1])
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: input channels and filter channels do not match");
    }

    int filter_row_cnt = this->filter_rows();
    int padded_row_cnt = this->in_feature_rows() + (this->padding_rows() * 2);
    while (filter_row_cnt < padded_row_cnt)
    {
        filter_row_cnt += this->stride_rows();
    }
    if (filter_row_cnt != padded_row_cnt)
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: filter/stride row combination does not fit input row count");
    }

    int filter_col_cnt = this->filter_cols();
    int padded_col_cnt = this->in_feature_cols() + (this->padding_cols() * 2);
    while (filter_col_cnt < padded_col_cnt)
    {
        filter_col_cnt += this->stride_cols();
    }
    if (filter_col_cnt != padded_col_cnt)
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: filter/stride column combination does not fit input column count");
    }

    if (this->filter_rows() < this->stride_rows())
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: filter row count less than stride row count");
    }

    if (this->filter_cols() < this->stride_cols())
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: filter column count less than stride column count");
    }
}

void Conv2d::reset_shape()
{
    if (this->input_shape() != this->default_n_shape_)
    {
        delete this->n_;
        this->n_ = NdArray::zeros(true, this->default_n_shape_);
    }
}

void Conv2d::summarize()
{
    std::string cls_name(typeid(*this).name());
    for (int i = cls_name.size(); i < 26; i++)
    {
        cls_name.push_back(' ');
    }

    printf("%s\t", cls_name.c_str());
    this->input_shape().print();

    if (this->padding_rows() > 0 || this->padding_cols() > 0)
    {
        printf(" (");
        this->padded_shape().print();
        printf(")");
    }

    printf(" -> ");
    this->output_shape().print();

    printf("\tFilter ");
    this->filter_shape().print();

    printf("\tPad (%d, %d)\tStride (%d, %d)", this->padding_rows(), this->padding_cols(), this->stride_rows(), this->stride_cols());
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
    return this->filter_shape()[0];
}

int Conv2d::filter_rows()
{
    return this->filter_shape()[2];
}

int Conv2d::filter_cols()
{
    return this->filter_shape()[3];
}

Shape Conv2d::filter_shape()
{
    return this->params_->weights()->shape();
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

Shape Conv2d::padded_shape()
{
    return Shape(this->batch_size(), this->channels(), this->default_n_shape_[2] + (this->padding_rows() * 2),
                 this->default_n_shape_[3] + (this->padding_cols() * 2));
}