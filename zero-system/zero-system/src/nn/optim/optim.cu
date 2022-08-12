#include "optim.cuh"

using namespace nn::optim;
using namespace nn::layer;

__global__ void k_sgd_weight_step(float *w, float *dw, int w_cnt, float lr, int batch_size)
{
    int w_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (w_elem_idx < w_cnt)
    {
        w[w_elem_idx] -= (lr * dw[w_elem_idx] / batch_size);
    }
}

__global__ void k_sgd_bias_step(float *b, float *db, int b_cnt, float lr, int batch_size)
{
    int b_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_elem_idx < b_cnt)
    {
        b[b_elem_idx] -= (lr * db[b_elem_idx] / batch_size);
    }
}

__global__ void k_sgd_momentum_weight_step(float *w, float *dw, float *vdw, int w_cnt, float lr, int batch_size, float momentum)
{
    int w_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (w_elem_idx < w_cnt)
    {
        vdw[w_elem_idx] = momentum * vdw[w_elem_idx] + (1.0f - momentum) * dw[w_elem_idx];
        w[w_elem_idx] -= (lr * vdw[w_elem_idx] / batch_size);
    }
}

__global__ void k_sgd_momentum_bias_step(float *b, float *db, float *vdb, int b_cnt, float lr, int batch_size, float momentum)
{
    int b_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_elem_idx < b_cnt)
    {
        vdb[b_elem_idx] = momentum * vdb[b_elem_idx] + (1.0f - momentum) * db[b_elem_idx];
        b[b_elem_idx] -= (lr * vdb[b_elem_idx] / batch_size);
    }
}

Optimizer::Optimizer(std::vector<Parameters *> model_params, float learning_rate)
{
    this->model_params_ = model_params;
    this->lr_ = learning_rate;
}

void Optimizer::summarize()
{
    std::string cls_name(typeid(*this).name());
    printf("%s", cls_name.c_str());

    size_t params_cnt = 0;
    for (Parameters *params : this->model_params_)
    {
        params_cnt += params->count();
    }

    printf("\n\tParameters: %zd\t\tLearning rate: %f", params_cnt, this->lr_);
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

        int w_cnt = w->count();
        int b_cnt = b->count();

        k_sgd_weight_step<<<w_cnt / CUDA_THREADS_PER_BLOCK + 1, CUDA_THREADS_PER_BLOCK>>>(w->data(), dw->data(), w_cnt, this->lr_, batch_size);
        k_sgd_bias_step<<<b_cnt / CUDA_THREADS_PER_BLOCK + 1, CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), b_cnt, this->lr_, batch_size);

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

        int w_cnt = w->count();
        int b_cnt = b->count();

        k_sgd_momentum_weight_step<<<w_cnt / CUDA_THREADS_PER_BLOCK + 1, CUDA_THREADS_PER_BLOCK>>>(w->data(), dw->data(), vdw->data(),
                                                                                                   w_cnt, this->lr_, batch_size, this->momentum_);
        k_sgd_momentum_bias_step<<<b_cnt / CUDA_THREADS_PER_BLOCK + 1, CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), vdb->data(),
                                                                                                 b_cnt, this->lr_, batch_size, this->momentum_);

        params->zero_grad();
    }
}