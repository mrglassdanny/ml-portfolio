#include "linear.cuh"

using namespace nn::layer;

__device__ float d_linear_sigmoid_evaluate(float val)
{
    return (1.0f / (1.0f + exp(-val)));
}

__device__ float d_linear_sigmoid_derive(float sigmoid_val)
{
    return (sigmoid_val) * (1.0f - sigmoid_val);
}

__device__ float d_linear_tanh_evaluate(float val)
{
    return ((exp(val) - exp(-val)) / (exp(val) + exp(-val)));
}

__device__ float d_linear_tanh_derive(float tanh_val)
{
    return (1.0f - (tanh_val * tanh_val));
}

__device__ float d_linear_relu_evaluate(float val)
{
    return val > 0.0f ? val : 0.0f;
}

__device__ float d_linear_relu_derive(float relu_val)
{
    return relu_val > 0.0f ? 1.0f : 0.0f;
}

__global__ void k_linear_evaluate(float *in, float *w, float *b, float *out,
                                  int batch_size, int in_cnt, int out_cnt, Activation activation)
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

        switch (activation)
        {
        case Activation::None:
            break;
        case Activation::Sigmoid:
            out[out_elem_idx] = d_linear_sigmoid_evaluate(out[out_elem_idx]);
            break;
        case Activation::Tanh:
            out[out_elem_idx] = d_linear_tanh_evaluate(out[out_elem_idx]);
            break;
        case Activation::ReLU:
            out[out_elem_idx] = d_linear_relu_evaluate(out[out_elem_idx]);
            break;
        default: // None
            break;
        }
    }
}

__global__ void k_linear_inc_param_derivatives(float *in, float *in_n, float *n, float *dw, float *db,
                                               int batch_size, int in_cnt, int n_cnt, int w_row_cnt, int w_col_cnt, Activation activation)
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

            switch (activation)
            {
            case Activation::None:
                break;
            case Activation::Sigmoid:
                in[batch_idx * in_cnt + in_idx] *= d_linear_sigmoid_derive(in_n[batch_idx * in_cnt + in_idx]);
                break;
            case Activation::Tanh:
                in[batch_idx * in_cnt + in_idx] *= d_linear_tanh_derive(in_n[batch_idx * in_cnt + in_idx]);
                break;
            case Activation::ReLU:
                in[batch_idx * in_cnt + in_idx] *= d_linear_relu_derive(in_n[batch_idx * in_cnt + in_idx]);
                break;
            default: // None
                break;
            }

            dw[w_elem_idx] += (in[batch_idx * in_cnt + in_idx] * n[batch_idx * n_cnt + n_idx]);

            if (w_row_idx == 0)
            {
                int b_elem_idx = w_col_idx;
                db[b_elem_idx] += in[batch_idx * in_cnt + in_idx];
            }
        }
    }
}

__global__ void k_linear_agg_derivatives(float *in, float *w, float *out, int batch_size, int in_cnt, int w_row_cnt, int w_col_cnt, int out_cnt)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx < out_cnt && batch_idx < batch_size)
    {
        int out_elem_idx = batch_idx * w_row_cnt + out_idx;
        int w_row_idx = out_idx;

        for (int in_idx = 0; in_idx < in_cnt; in_idx++)
        {
            int w_col_idx = in_idx;
            out[out_elem_idx] += (in[batch_idx * in_cnt + in_idx] * w[w_row_idx * w_col_cnt + w_col_idx]);
        }
    }
}

Linear::Linear(Shape in_shape, Shape out_shape, Activation activation)
{
    this->n_ = new NdArray(true, in_shape);
    this->default_n_shape_ = in_shape;

    int in_cnt = (in_shape.dims_size() / this->batch_size());
    int out_cnt = (out_shape.dims_size() / this->batch_size());

    this->params_ = new Parameters(Shape(in_cnt, out_cnt), Shape(out_cnt), in_cnt, out_cnt);

    this->activation_ = activation;
}

void Linear::evaluate(NdArray *out)
{
    int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (this->out_features() / CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();

    k_linear_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), b->data(), out->data(),
                                                 this->batch_size(), this->in_features(), this->out_features(), this->activation_);
}

NdArray *Linear::derive(NdArray *in, NdArray *in_n)
{
    NdArray *n = this->n_;
    NdArray *w = this->params_->weights();
    NdArray *b = this->params_->biases();
    NdArray *dw = this->params_->weight_gradients();
    NdArray *db = this->params_->bias_gradients();

    {
        int grid_row_cnt = (this->weight_rows() / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->weight_cols() / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_linear_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), in_n->data(), n->data(), dw->data(), db->data(),
                                                                  this->batch_size(), this->out_features(), this->in_features(),
                                                                  this->weight_rows(), this->weight_cols(), this->activation_);
    }

    NdArray *out = NdArray::zeros(true, this->input_shape());

    {
        int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->in_features() / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_linear_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), out->data(),
                                                            this->batch_size(), this->out_features(), this->weight_rows(), this->weight_cols(), this->in_features());
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
    return Shape(this->batch_size(), this->weight_cols());
}

void Linear::validate()
{
    if (this->input_shape().num_dims() < 2)
    {
        THROW_ERROR("LINEAR LAYER VALIDATION FAILED: invalid input shape");
    }

    if (this->output_shape().num_dims() != 2)
    {
        THROW_ERROR("LINEAR LAYER VALIDATION FAILED: invalid output shape");
    }
}

void Linear::summarize()
{
    Layer::summarize();

    printf("\tActivation (");
    switch (this->activation_)
    {
    case Activation::None:
        printf("None");
        break;
    case Activation::Sigmoid:
        printf("Sigmoid");
        break;
    case Activation::Tanh:
        printf("Tanh");
        break;
    case Activation::ReLU:
        printf("ReLU");
        break;
    default: // None
        break;
    }
    printf(")");
}

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
