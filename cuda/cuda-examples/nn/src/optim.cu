#include "optim.cuh"

using namespace nn::optim;
using namespace nn::layer;

__global__ void k_sgd_step(float *w, float *b, float *dw, float *db, int w_row_cnt, int w_col_cnt, float lr, int batch_size)
{
    int w_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int w_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_col_idx < w_row_cnt && w_row_idx < w_col_cnt)
    {
        int w_elem_idx = w_row_idx * w_col_cnt + w_col_idx;
        w[w_elem_idx] -= (lr * dw[w_elem_idx] / batch_size);

        if (w_row_idx == 0)
        {
            int b_elem_idx = w_col_idx;
            b[b_elem_idx] -= (lr * db[b_elem_idx] / batch_size);
        }
    }
}

__global__ void k_sgd_momentum_step(float *w, float *b, float *dw, float *db, float *vdw, float *vdb, int w_row_cnt, int w_col_cnt, float lr, int batch_size, float momentum)
{
    int w_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int w_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_col_idx < w_row_cnt && w_row_idx < w_col_cnt)
    {
        int w_elem_idx = w_row_idx * w_col_cnt + w_col_idx;
        vdw[w_elem_idx] = momentum * vdw[w_elem_idx] + (1.0f - momentum) * dw[w_elem_idx];
        w[w_elem_idx] -= (lr * vdw[w_elem_idx] / batch_size);

        if (w_row_idx == 0)
        {
            int b_elem_idx = w_col_idx;
            vdb[b_elem_idx] = momentum * vdb[b_elem_idx] + (1.0f - momentum) * db[b_elem_idx];
            b[b_elem_idx] -= (lr * vdb[b_elem_idx] / batch_size);
        }
    }
}

Optimizer::Optimizer(std::vector<Parameters *> model_params, float learning_rate)
{
    this->model_params_ = model_params;
    this->lr_ = learning_rate;
}

SGD::SGD(std::vector<Parameters *> model_params, float learning_rate)
    : Optimizer(model_params, learning_rate)
{
}

void SGD::step(int batch_size)
{
    for (Parameters *params : this->model_params_)
    {
        NdArray *w = params->weights();
        NdArray *b = params->biases();
        NdArray *dw = params->weight_gradients();
        NdArray *db = params->bias_gradients();

        int grid_row_cnt = (w->rows() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (w->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_sgd_step<<<grid_dims, block_dims>>>(w->data(), b->data(), dw->data(), db->data(),
                                              w->rows(), w->cols(), this->lr_, batch_size);

        params->zero_grad();
    }
}

SGDMomentum::SGDMomentum(std::vector<Parameters *> model_params, float learning_rate, float momentum)
    : Optimizer(model_params, learning_rate)
{
    this->momentum_ = momentum;

    for (Parameters *params : model_params)
    {
        this->vdws_.push_back(new NdArray(true, params->weight_gradients()->shape()));
        this->vdbs_.push_back(new NdArray(true, params->bias_gradients()->shape()));
    }
}

SGDMomentum::~SGDMomentum()
{
    for (int i = 0; i < this->vdws_.size(); i++)
    {
        delete this->vdws_[i];
        delete this->vdbs_[i];
    }
}

void SGDMomentum::step(int batch_size)
{
    for (int i = 0; i < this->model_params_.size(); i++)
    {
        Parameters *params = this->model_params_[i];

        NdArray *w = params->weights();
        NdArray *b = params->biases();
        NdArray *dw = params->weight_gradients();
        NdArray *db = params->bias_gradients();
        NdArray *vdw = this->vdws_[i];
        NdArray *vdb = this->vdbs_[i];

        int grid_row_cnt = (w->rows() / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (w->cols() / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_sgd_momentum_step<<<grid_dims, block_dims>>>(w->data(), b->data(), dw->data(), db->data(), vdw->data(), vdb->data(),
                                                       w->rows(), w->cols(), this->lr_, batch_size, this->momentum_);

        params->zero_grad();
    }
}