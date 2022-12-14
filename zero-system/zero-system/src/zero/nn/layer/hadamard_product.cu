#include "hadamard_product.cuh"

using namespace zero::core;
using namespace zero::nn::layer;

__global__ void k_hadamard_product_evaluate(float *in, float *w, float *out, int batch_size, int channel_cnt, int row_cnt, int col_cnt,
                                            int filter_cnt)
{
    int f_r_c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int filter_idx = f_r_c_idx / (row_cnt * col_cnt);
    int row_idx = (f_r_c_idx - (filter_idx * (row_cnt * col_cnt))) / col_cnt;
    int col_idx = f_r_c_idx % col_cnt;

    if (filter_idx < filter_cnt && row_idx < row_cnt && col_idx < col_cnt && batch_idx < batch_size)
    {
        int in_cnt = row_cnt * col_cnt;
        int w_cnt = in_cnt;
        int out_cnt = in_cnt;

        float *l_in = &in[(batch_idx * channel_cnt * in_cnt)];
        float *l_w = &w[(filter_idx * channel_cnt * w_cnt)];
        float *l_out = &out[((batch_idx * filter_cnt * out_cnt) + (filter_idx * out_cnt))];

        for (int channel_idx = 0; channel_idx < channel_cnt; channel_idx++)
        {
            l_out[row_idx * col_cnt + col_idx] +=
                (l_in[(channel_idx * in_cnt) + (row_idx * col_cnt) + col_idx] *
                 l_w[(channel_idx * w_cnt) + (row_idx * col_cnt) + col_idx]);
        }
    }
}

__global__ void k_hadamard_product_inc_param_derivatives(float *in, float *in_n, float *n, float *dw, int batch_size, int channel_cnt, int filter_cnt,
                                                         int row_cnt, int col_cnt, int cnt)
{
    int c_r_c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int filter_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int channel_idx = c_r_c_idx / cnt;
    int row_idx = (c_r_c_idx - (channel_idx * cnt)) / col_cnt;
    int col_idx = c_r_c_idx % col_cnt;

    if (channel_idx < channel_cnt && row_idx < row_cnt && col_idx < col_cnt && filter_idx < filter_cnt)
    {
        int w_elem_idx = (filter_idx * channel_cnt * cnt) + (channel_idx * cnt) + (row_idx * col_cnt) + col_idx;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            int in_elem_idx = (batch_idx * filter_cnt * cnt) + (filter_idx * cnt) + (row_idx * col_cnt) + col_idx;

            dw[w_elem_idx] +=
                (in[in_elem_idx] * n[(batch_idx * channel_cnt * cnt) + (channel_idx * cnt) + (row_idx * col_cnt) + col_idx]);
        }
    }
}

__global__ void k_hadamard_product_agg_derivatives(float *in, float *w, float *out, int batch_size, int channel_cnt, int filter_cnt,
                                                   int row_cnt, int col_cnt, int cnt)
{
    int f_r_c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int filter_idx = f_r_c_idx / cnt;
    int row_idx = (f_r_c_idx - (filter_idx * cnt)) / col_cnt;
    int col_idx = f_r_c_idx % col_cnt;

    if (filter_idx < filter_cnt && row_idx < row_cnt && col_idx < col_cnt && batch_idx < batch_size)
    {
        int in_elem_idx = (batch_idx * filter_cnt * cnt) + (filter_idx * cnt) + (row_idx * col_cnt) + col_idx;

        for (int channel_idx = 0; channel_idx < channel_cnt; channel_idx++)
        {
            int out_elem_idx = (batch_idx * channel_cnt * cnt) + (channel_idx * cnt) + (row_idx * col_cnt) + col_idx;

            atomicAdd(&out[out_elem_idx], (in[in_elem_idx] * w[(filter_idx * channel_cnt * cnt) + (channel_idx * cnt) + (row_idx * col_cnt) + col_idx]));
        }
    }
}

HadamardProduct::HadamardProduct(bool shared_params, Shape in_shape, int filter_cnt, Activation *activation, Initializer *initializer)
    : Learnable(shared_params)
{
    this->n_ = new Tensor(true, in_shape);
    this->dn_ = Tensor::zeros(true, in_shape);
    this->activation_ = activation;

    if (!this->shared_params_)
    {
        this->params_ = new Parameters(Shape(filter_cnt, this->channels(), this->rows(), this->cols()),
                                       Shape(1), this->rows(), this->cols(), initializer);
    }
}

void HadamardProduct::evaluate(Tensor *out)
{
    int grid_row_cnt = (this->batch_size() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = ((this->filters() * this->rows() * this->cols()) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    Tensor *n = this->n_;
    Tensor *w = this->params_->weights();

    k_hadamard_product_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), out->data(), this->batch_size(), this->channels(), this->rows(), this->cols(),
                                                           this->filters());

    if (this->activation_ != nullptr)
    {
        this->activation_->evaluate(out, this->batch_size(), this->out_features());
    }
}

void HadamardProduct::derive(Tensor *in, Tensor *in_n)
{
    Tensor *n = this->n_;
    Tensor *dn = this->dn_;
    Tensor *w = this->params_->weights();
    Tensor *b = this->params_->biases();
    Tensor *dw = this->params_->weight_gradients();

    if (this->activation_ != nullptr)
    {
        this->activation_->derive(in, in_n, this->batch_size(), this->out_features());
    }

    {
        int grid_row_cnt = (this->filters() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = ((this->channels() * this->rows() * this->cols()) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

        k_hadamard_product_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), in_n->data(), n->data(), dw->data(), this->batch_size(), this->channels(), this->filters(),
                                                                            this->rows(), this->cols(), (this->rows() * this->cols()));
    }

    {
        int grid_row_cnt = (this->batch_size() / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = ((this->filters() * this->rows() * this->cols()) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

        k_hadamard_product_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), dn->data(), this->batch_size(), this->channels(), this->filters(),
                                                                      this->rows(), this->cols(), (this->rows() * this->cols()));
    }
}

Shape HadamardProduct::input_shape()
{
    return this->n_->shape();
}

Shape HadamardProduct::output_shape()
{
    return Shape(this->batch_size(), this->filters(), this->rows(), this->cols());
}

Layer *HadamardProduct::copy()
{
    auto lyr = new HadamardProduct(true, this->input_shape(), this->filters(), this->activation_->copy(), nullptr);
    lyr->share_parameters(this->params_);
    return lyr;
}

void HadamardProduct::validate()
{
    if (this->input_shape().num_dims() != 4)
    {
        ZERO_CORE_THROW_ERROR("HADAMARD_PRODUCT LAYER VALIDATION FAILED: invalid input shape");
    }
}

void HadamardProduct::summarize()
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

    if (this->activation_ != nullptr)
    {
        this->activation_->summarize();
    }
}

int HadamardProduct::channels()
{
    return this->n_->shape()[1];
}

int HadamardProduct::rows()
{
    return this->n_->shape()[2];
}

int HadamardProduct::cols()
{
    return this->n_->shape()[3];
}

int HadamardProduct::filters()
{
    return this->params_->weights()->shape()[0];
}