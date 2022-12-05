#include "matrix_product.cuh"

using namespace zero::core;
using namespace zero::nn::layer;

__global__ void k_matrix_product_evaluate(float *in, float *w, float *out,
                                          int row_cnt, int col_cnt, int cnt,
                                          int batch_size, int channel_cnt, int filter_cnt)
{
    int r_c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_c_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int batch_idx = b_c_idx / channel_cnt;
    int channel_idx = b_c_idx % channel_cnt;

    int row_idx = r_c_idx / col_cnt;
    int col_idx = r_c_idx % col_cnt;

    if (col_idx < col_cnt && row_idx < row_cnt && channel_idx < channel_cnt && batch_idx < batch_size)
    {
        for (int filter_idx = 0; filter_idx < filter_cnt; filter_idx++)
        {
            int out_elem_idx = (batch_idx * filter_cnt * cnt) + (filter_idx * cnt) + (row_idx * col_cnt) + col_idx;

            for (int in_col_idx = 0; in_col_idx < col_cnt; in_col_idx++)
            {
                int in_idx = (batch_idx * channel_cnt * cnt) + (channel_idx * cnt) + (row_idx * col_cnt) + in_col_idx;
                int w_idx = (filter_idx * channel_cnt * cnt) + (channel_idx * cnt) + (in_col_idx * col_cnt) + col_idx;

                atomicAdd(&out[out_elem_idx], (in[in_idx] * w[w_idx]));
            }
        }
    }
}

__global__ void k_matrix_product_inc_param_derivatives(float *in, float *in_n, float *n, float *dw,
                                                       int row_cnt, int col_cnt, int cnt,
                                                       int filter_cnt, int batch_size, int channel_cnt)
{
    int r_c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int filter_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int row_idx = r_c_idx / col_cnt;
    int col_idx = r_c_idx % col_cnt;

    if (col_idx < col_cnt && row_idx < row_cnt && filter_idx < filter_cnt)
    {
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            for (int channel_idx = 0; channel_idx < channel_cnt; channel_idx++)
            {
                int w_elem_idx = (filter_idx * channel_cnt * cnt) + (channel_idx * cnt) + (row_idx * col_cnt) + col_idx;

                for (int in_col_idx = 0; in_col_idx < col_cnt; in_col_idx++)
                {
                    int n_idx = (batch_idx * channel_cnt * cnt) + (channel_idx * cnt) + (in_col_idx * col_cnt) + row_idx;
                    int in_idx = (batch_idx * filter_cnt * cnt) + (filter_idx * cnt) + (in_col_idx * col_cnt) + col_idx;

                    dw[w_elem_idx] += (in[in_idx] * n[n_idx]);
                }
            }
        }
    }
}

__global__ void k_matrix_product_agg_derivatives(float *in, float *w, float *out,
                                                 int row_cnt, int col_cnt, int cnt,
                                                 int batch_size, int channel_cnt, int filter_cnt)
{
    int r_c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_c_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int batch_idx = b_c_idx / channel_cnt;
    int channel_idx = b_c_idx % channel_cnt;

    int row_idx = r_c_idx / col_cnt;
    int col_idx = r_c_idx % col_cnt;

    if (col_idx < col_cnt && row_idx < row_cnt && channel_idx < channel_cnt && batch_idx < batch_size)
    {
        for (int filter_idx = 0; filter_idx < filter_cnt; filter_idx++)
        {
            int out_elem_idx = (batch_idx * channel_cnt * cnt) + (channel_idx * cnt) + (row_idx * col_cnt) + col_idx;

            for (int in_col_idx = 0; in_col_idx < col_cnt; in_col_idx++)
            {
                int in_idx = (batch_idx * filter_cnt * cnt) + (filter_idx * cnt) + (row_idx * col_cnt) + in_col_idx;
                int w_idx = (filter_idx * channel_cnt * cnt) + (channel_idx * cnt) + (col_idx * col_cnt) + in_col_idx;

                out[out_elem_idx] += (in[in_idx] * w[w_idx]);
            }
        }
    }
}

MatrixProduct::MatrixProduct(bool shared_params, Shape in_shape, int filter_cnt, ActivationType activation)
    : Learnable(shared_params)
{

    this->n_ = new Tensor(true, in_shape);
    this->dn_ = new Tensor(true, in_shape);

    if (!this->shared_params_)
    {
        this->params_ = new Parameters(Shape(filter_cnt, this->channels(), this->rows(), this->cols()),
                                       Shape(1), this->rows(), this->cols());
    }

    this->activation_ = activation;
}

void MatrixProduct::evaluate(Tensor *out)
{
    int grid_row_cnt = ((this->batch_size() * this->channels()) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = ((this->rows() * this->cols()) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    Tensor *n = this->n_;
    Tensor *w = this->params_->weights();

    k_matrix_product_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), out->data(),
                                                         this->rows(), this->cols(), this->rows() * this->cols(),
                                                         this->batch_size(), this->channels(), this->filters());

    Activation::evaluate(out, this->batch_size(), this->out_features(), this->activation_);
}

void MatrixProduct::derive(Tensor *in, Tensor *in_n)
{
    Tensor *n = this->n_;
    Tensor *dn = this->dn_;
    Tensor *w = this->params_->weights();
    Tensor *b = this->params_->biases();
    Tensor *dw = this->params_->weight_gradients();
    Tensor *db = this->params_->bias_gradients();

    Activation::derive(in, in_n, this->batch_size(), this->out_features(), this->activation_);

    {
        int grid_row_cnt = (this->filters() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = ((this->rows() * this->cols()) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

        k_matrix_product_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), in_n->data(), n->data(), dw->data(),
                                                                          this->rows(), this->cols(), this->rows() * this->cols(),
                                                                          this->filters(), this->batch_size(), this->channels());
    }

    {
        int grid_row_cnt = ((this->batch_size() * this->channels()) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = ((this->rows() * this->cols()) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

        k_matrix_product_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), dn->data(),
                                                                    this->rows(), this->cols(), this->rows() * this->cols(),
                                                                    this->batch_size(), this->channels(), this->filters());
    }
}

Shape MatrixProduct::input_shape()
{
    return this->n_->shape();
}

Shape MatrixProduct::output_shape()
{
    return Shape(this->batch_size(), this->filters(), this->rows(), this->cols());
}

Layer *MatrixProduct::copy()
{
    auto lyr = new MatrixProduct(true, this->input_shape(), this->filters(), this->activation_);
    lyr->share_parameters(this->params_);
    return lyr;
}

void MatrixProduct::validate()
{
    if (this->input_shape().num_dims() != 4)
    {
        ZERO_CORE_THROW_ERROR("MATRIX_PRODUCT LAYER VALIDATION FAILED: invalid input shape");
    }
}

void MatrixProduct::summarize()
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

    printf("\tFilters: %d\t", this->filters());

    Activation::summarize(this->activation_);
}

int MatrixProduct::channels()
{
    return this->n_->shape()[1];
}

int MatrixProduct::rows()
{
    return this->n_->shape()[2];
}

int MatrixProduct::cols()
{
    return this->n_->shape()[3];
}

int MatrixProduct::filters()
{
    return this->params_->weights()->shape()[0];
}