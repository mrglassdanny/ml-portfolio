#include "enhanced_residual.cuh"

using namespace nn::layer;

__global__ void k_enhanced_residual_evaluate(float *in, float *w, float *b, float *out,
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

__global__ void k_enhanced_residual_inc_param_derivatives(float *in, float *in_n, float *n, float *dw, float *db,
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

__global__ void k_enhanced_residual_agg_derivatives(float *in, float *w, float *out, int batch_size, int in_cnt, int w_row_cnt, int w_col_cnt, int out_cnt)
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

EnhancedResidual::EnhancedResidual(Shape in_shape, Shape out_shape, ActivationType activation)
{
    this->n_ = new NdArray(true, in_shape);
    this->dn_ = new NdArray(true, in_shape);

    int in_cnt = (in_shape.dims_size() / this->batch_size());
    int out_cnt = (out_shape.dims_size() / this->batch_size());

    this->params_ = new Parameters(Shape(in_cnt, out_cnt), Shape(out_cnt), in_cnt, out_cnt);

    this->activation_ = activation;
}

EnhancedResidual::~EnhancedResidual()
{
    for (Parameters *rp : this->residual_params_)
    {
        delete rp;
    }
}

void EnhancedResidual::evaluate_residual(NdArray *out, int idx)
{
    int out_feature_cnt = out->dims_size() / this->batch_size();

    int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (out_feature_cnt / CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

    NdArray *n = this->n_;
    NdArray *w = this->residual_params_[idx]->weights();
    NdArray *b = this->residual_params_[idx]->biases();

    k_enhanced_residual_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), b->data(), out->data(),
                                                            this->batch_size(), this->in_features(), out_feature_cnt);
}

void EnhancedResidual::derive_residual(NdArray *in, NdArray *in_n, int idx)
{
    NdArray *n = this->n_;
    NdArray *w = this->residual_params_[idx]->weights();
    NdArray *b = this->residual_params_[idx]->biases();
    NdArray *dw = this->residual_params_[idx]->weight_gradients();
    NdArray *db = this->residual_params_[idx]->bias_gradients();

    int w_row_cnt = w->shape()[0];
    int w_col_cnt = w->shape()[1];
    int out_cnt = w_col_cnt;

    {
        int grid_row_cnt = (w_row_cnt / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (w_col_cnt / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_enhanced_residual_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), in_n->data(), n->data(), dw->data(), db->data(),
                                                                             this->batch_size(), out_cnt, this->in_features(),
                                                                             w_row_cnt, w_col_cnt);
    }

    {
        int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->in_features() / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_enhanced_residual_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), this->dn_->data(),
                                                                       this->batch_size(), out_cnt, w_row_cnt, w_col_cnt, this->in_features());
    }
}

void EnhancedResidual::compile(std::vector<Layer *> layers, int my_idx)
{
    int fan_in = this->in_features();

    for (int i = my_idx + 1; i < layers.size(); i++)
    {
        Layer *lyr = layers[i];
        int fan_out = lyr->out_features();

        this->residual_params_.push_back(new Parameters(Shape(fan_in, fan_out), Shape(fan_out), fan_in, fan_out));
    }
}

std::vector<Parameters *> EnhancedResidual::residual_parameters()
{
    return this->residual_params_;
}
