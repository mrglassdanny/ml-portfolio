#include "matrix_product.cuh"

using namespace zero::core;
using namespace zero::nn::layer;

__global__ void k_matrix_product_evaluate(float *in, float *w, float *out,
                                          int row_cnt, int col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        int out_elem_idx = row_idx * col_cnt + col_idx;

        for (int in_col_idx = 0; in_col_idx < col_cnt; in_col_idx++)
        {
            out[out_elem_idx] += (in[row_idx * col_cnt + in_col_idx] * w[in_col_idx * col_cnt + col_idx]);
        }
    }
}

__global__ void k_matrix_product_inc_param_derivatives(float *in, float *in_n, float *n, float *dw,
                                                       int row_cnt, int col_cnt, int cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        int w_elem_idx = row_idx * col_cnt + col_idx;
        int n_idx = row_idx;
        int in_idx = col_idx;

        for (int in_col_idx = 0; in_col_idx < col_cnt; in_col_idx++)
        {
            dw[w_elem_idx] += (in[in_col_idx * col_cnt + in_idx] * n[in_col_idx * col_cnt + n_idx]);
        }
    }
}

__global__ void k_matrix_product_agg_derivatives(float *in, float *w, float *out, int row_cnt, int col_cnt)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < col_cnt && row_idx < row_cnt)
    {
        int out_elem_idx = row_idx * col_cnt + col_idx;
        int w_row_idx = col_idx;

        for (int in_col_idx = 0; in_col_idx < col_cnt; in_col_idx++)
        {
            out[out_elem_idx] += (in[row_idx * col_cnt + in_col_idx] * w[col_idx * col_cnt + in_col_idx]);
        }
    }
}

MatrixProduct::MatrixProduct(Shape in_shape, int filter_cnt, ActivationType activation)
{
    this->n_ = new Tensor(true, in_shape);
    this->dn_ = new Tensor(true, in_shape);
    this->params_ = new Parameters(Shape(filter_cnt, this->channels(), this->rows(), this->cols()),
                                   Shape(filter_cnt, this->channels()), this->rows(), this->cols());

    this->activation_ = activation;
}

void MatrixProduct::evaluate(Tensor *out)
{
    int grid_row_cnt = (this->rows() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->cols() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    Tensor *n = this->n_;
    Tensor *w = this->params_->weights();

    k_matrix_product_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), out->data(),
                                                         this->rows(), this->cols());

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
        int grid_row_cnt = (this->rows() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->cols() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

        k_matrix_product_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), in_n->data(), n->data(), dw->data(),
                                                                          this->rows(), this->cols(), this->rows() * this->cols());
    }

    {
        int grid_row_cnt = (this->batch_size() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->in_features() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

        k_matrix_product_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), dn->data(),
                                                                    this->rows(), this->cols());
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