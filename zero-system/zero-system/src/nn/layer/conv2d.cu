#include "conv2d.cuh"

using namespace nn::layer;

__global__ void k_conv2d_evaluate(float *in, float *w, float *b, float *out, int batch_size, int channel_cnt, int in_row_cnt, int in_col_cnt,
                                  int filter_cnt, int filter_row_cnt, int filter_col_cnt, int out_row_cnt, int out_col_cnt,
                                  int stride_row_cnt, int stride_col_cnt)
{
    int f_or_oc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int filter_idx = f_or_oc_idx / (out_row_cnt * out_col_cnt);
    int out_row_idx = (f_or_oc_idx - (filter_idx * (out_row_cnt * out_col_cnt))) / out_col_cnt;
    int out_col_idx = f_or_oc_idx % out_col_cnt;

    if (filter_idx < filter_cnt && out_row_idx < out_row_cnt && out_col_idx < out_col_cnt && batch_idx < batch_size)
    {
        int in_cnt = in_row_cnt * in_col_cnt;
        int w_cnt = filter_row_cnt * filter_col_cnt;
        int out_cnt = out_row_cnt * out_col_cnt;

        float *l_in = &in[(batch_idx * channel_cnt * in_cnt)];
        float *l_w = &w[(filter_idx * channel_cnt * w_cnt)];
        float *l_b = &b[(filter_idx * channel_cnt)];
        float *l_out = &out[((batch_idx * filter_cnt * out_cnt) + (filter_idx * out_cnt))];

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

__global__ void k_conv2d_inc_param_derivatives(float *in, float *in_n, float *n, float *dw, float *db, int batch_size, int channel_cnt, int filter_cnt,
                                               int in_row_cnt, int in_col_cnt, int in_cnt, int n_row_cnt, int n_col_cnt, int n_cnt, int w_row_cnt, int w_col_cnt,
                                               int w_cnt, int stride_row_cnt, int stride_col_cnt)
{
    int c_wr_wc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int filter_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int channel_idx = c_wr_wc_idx / w_cnt;
    int w_row_idx = (c_wr_wc_idx - (channel_idx * w_cnt)) / w_col_cnt;
    int w_col_idx = c_wr_wc_idx % w_col_cnt;

    if (channel_idx < channel_cnt && w_row_idx < w_row_cnt && w_col_idx < w_col_cnt && filter_idx < filter_cnt)
    {
        int w_elem_idx = (filter_idx * channel_cnt * w_cnt) + (channel_idx * w_cnt) + (w_row_idx * w_col_cnt + w_col_idx);
        int b_elem_idx = filter_idx * channel_cnt + channel_idx;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            for (int in_row_idx = 0; in_row_idx < in_row_cnt; in_row_idx++)
            {
                for (int in_col_idx = 0; in_col_idx < in_col_cnt; in_col_idx++)
                {
                    int in_elem_idx = (batch_idx * filter_cnt * in_cnt) + (filter_idx * in_cnt) + (in_row_idx * in_col_cnt + in_col_idx);

                    dw[w_elem_idx] +=
                        (in[in_elem_idx] *
                         n[(batch_idx * channel_cnt * n_cnt) + (channel_idx * n_cnt) + ((w_row_idx + (in_row_idx * stride_row_cnt)) * n_col_cnt + (w_col_idx + (in_col_idx * stride_col_cnt)))]);

                    if (w_row_idx == 0 && w_col_idx == 0)
                    {
                        db[b_elem_idx] += in[in_elem_idx];
                    }
                }
            }
        }
    }
}

__global__ void k_conv2d_agg_derivatives(float *in, float *w, float *out, int batch_size, int channel_cnt, int filter_cnt,
                                         int in_row_cnt, int in_col_cnt, int in_cnt, int w_row_cnt, int w_col_cnt, int w_cnt, int out_row_cnt, int out_col_cnt, int out_cnt,
                                         int stride_row_cnt, int stride_col_cnt)
{
    int f_ir_ic_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int filter_idx = f_ir_ic_idx / in_cnt;
    int in_row_idx = (f_ir_ic_idx - (filter_idx * in_cnt)) / in_col_cnt;
    int in_col_idx = f_ir_ic_idx % in_col_cnt;

    if (filter_idx < filter_cnt && in_row_idx < in_row_cnt && in_col_idx < in_col_cnt && batch_idx < batch_size)
    {
        int in_elem_idx = (batch_idx * filter_cnt * in_cnt) + (filter_idx * in_cnt) + (in_row_idx * in_col_cnt + in_col_idx);

        for (int channel_idx = 0; channel_idx < channel_cnt; channel_idx++)
        {
            for (int w_row_idx = 0; w_row_idx < w_row_cnt; w_row_idx++)
            {
                for (int w_col_idx = 0; w_col_idx < w_col_cnt; w_col_idx++)
                {
                    int out_elem_idx = (batch_idx * channel_cnt * out_cnt) + (channel_idx * out_cnt) + ((w_row_idx + (in_row_idx * stride_row_cnt)) * out_col_cnt + (w_col_idx + (in_col_idx * stride_col_cnt)));

                    atomicAdd(&out[out_elem_idx], (in[in_elem_idx] * w[(filter_idx * channel_cnt * w_cnt) + (channel_idx * w_cnt) + (w_row_idx * w_col_cnt + w_col_idx)]));
                }
            }
        }
    }
}

Conv2d::Conv2d(Shape in_shape, Shape filter_shape, Stride stride, ActivationType activation)
{
    this->n_ = new Tensor(true, in_shape);
    this->dn_ = new Tensor(true, in_shape);
    this->params_ = new Parameters(filter_shape, Shape(filter_shape[0], filter_shape[1]), this->in_feature_rows(), this->in_feature_cols());
    this->stride_ = stride;

    this->out_row_cnt_ = ((this->in_feature_rows() - this->filter_rows()) / this->stride_rows()) + 1;
    this->out_col_cnt_ = ((this->in_feature_cols() - this->filter_cols()) / this->stride_cols()) + 1;

    this->activation_ = activation;
}

void Conv2d::evaluate(Tensor *out)
{
    int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = ((this->filters() * this->out_feature_rows() * this->out_feature_cols()) / CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

    Tensor *n = this->n_;
    Tensor *w = this->params_->weights();
    Tensor *b = this->params_->biases();

    k_conv2d_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), b->data(), out->data(), this->batch_size(), this->channels(), this->in_feature_rows(), this->in_feature_cols(),
                                                 this->filters(), this->filter_rows(), this->filter_cols(), this->out_feature_rows(), this->out_feature_cols(),
                                                 this->stride_rows(), this->stride_cols());

    Activation::evaluate(out, this->batch_size(), this->out_features(), this->activation_);
}

void Conv2d::derive(Tensor *in, Tensor *in_n)
{
    Tensor *n = this->n_;
    Tensor *dn = this->dn_;
    Tensor *w = this->params_->weights();
    Tensor *b = this->params_->biases();
    Tensor *dw = this->params_->weight_gradients();
    Tensor *db = this->params_->bias_gradients();

    Activation::derive(in, in_n, this->batch_size(), this->out_features(), this->activation_);

    {
        int grid_row_cnt = (this->filters() / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = ((this->channels() * this->filter_rows() * this->filter_cols()) / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_conv2d_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), in_n->data(), n->data(), dw->data(), db->data(), this->batch_size(), this->channels(), this->filters(),
                                                                  this->out_feature_rows(), this->out_feature_cols(), (this->out_feature_rows() * this->out_feature_cols()),
                                                                  this->in_feature_rows(), this->in_feature_cols(), (this->in_feature_rows() * this->in_feature_cols()),
                                                                  this->filter_rows(), this->filter_cols(), (this->filter_rows() * this->filter_cols()),
                                                                  this->stride_rows(), this->stride_cols());
    }

    {
        int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = ((this->filters() * this->out_feature_rows() * this->out_feature_cols()) / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_conv2d_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), dn->data(), this->batch_size(), this->channels(), this->filters(),
                                                            this->out_feature_rows(), this->out_feature_cols(), (this->out_feature_rows() * this->out_feature_cols()),
                                                            this->filter_rows(), this->filter_cols(), (this->filter_rows() * this->filter_cols()),
                                                            this->in_feature_rows(), this->in_feature_cols(), (this->in_feature_rows() * this->in_feature_cols()),
                                                            this->stride_rows(), this->stride_cols());
    }
}

Shape Conv2d::input_shape()
{
    return this->n_->shape();
}

Shape Conv2d::output_shape()
{
    return Shape(this->batch_size(), this->filters(), this->out_row_cnt_, this->out_col_cnt_);
}

void Conv2d::validate()
{
    if (this->input_shape().num_dims() != 4)
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: invalid input shape");
    }

    if (this->filter_shape().num_dims() != 4)
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: invalid filter shape");
    }

    if (this->input_shape()[1] != this->filter_shape()[1])
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: input channels and filter channels do not match");
    }

    int filter_row_test = this->filter_rows();
    while (filter_row_test < this->in_feature_rows())
    {
        filter_row_test += this->stride_rows();
    }
    if (filter_row_test != this->in_feature_rows())
    {
        THROW_ERROR("CONV2D LAYER VALIDATION FAILED: filter/stride row combination does not fit input row count");
    }

    int filter_col_test = this->filter_cols();
    while (filter_col_test < this->in_feature_cols())
    {
        filter_col_test += this->stride_cols();
    }
    if (filter_col_test != this->in_feature_cols())
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

void Conv2d::summarize()
{
    std::string cls_name(typeid(*this).name());
    for (int i = cls_name.size(); i < 26; i++)
    {
        cls_name.push_back(' ');
    }

    printf("%s\t", cls_name.c_str());

    this->input_shape().print_pad(16, true);

    printf(" -> ");
    this->output_shape().print_pad(16, false);

    printf("\tFilter (");
    this->filter_shape().print();
    printf(")");

    printf("\tStride (%d, %d)\t", this->stride_rows(), this->stride_cols());

    Activation::summarize(this->activation_);
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
