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

__global__ void k_sgd_momentum_weight_step(float *w, float *dw, float *vdw, int w_cnt, float lr, float beta1, int batch_size, int iter)
{
    int w_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (w_elem_idx < w_cnt)
    {
        vdw[w_elem_idx] = beta1 * vdw[w_elem_idx] + (1.0f - beta1) * dw[w_elem_idx];

        vdw[w_elem_idx] /= (1.0f - pow(beta1, iter));

        w[w_elem_idx] -= (lr * vdw[w_elem_idx] / batch_size);
    }
}

__global__ void k_sgd_momentum_bias_step(float *b, float *db, float *vdb, int b_cnt, float lr, float beta1, int batch_size, int iter)
{
    int b_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_elem_idx < b_cnt)
    {
        vdb[b_elem_idx] = beta1 * vdb[b_elem_idx] + (1.0f - beta1) * db[b_elem_idx];

        vdb[b_elem_idx] /= (1.0f - pow(beta1, iter));

        b[b_elem_idx] -= (lr * vdb[b_elem_idx] / batch_size);
    }
}

__global__ void k_adam_weight_step(float *w, float *dw, float *vdw, float *sdw, int w_cnt, float lr, float beta1, float beta2, int batch_size, int iter)
{
    int w_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (w_elem_idx < w_cnt)
    {
        vdw[w_elem_idx] = beta1 * vdw[w_elem_idx] + (1.0f - beta1) * dw[w_elem_idx];
        sdw[w_elem_idx] = beta2 * sdw[w_elem_idx] + (1.0f - beta2) * (dw[w_elem_idx] * dw[w_elem_idx]);

        vdw[w_elem_idx] /= (1.0f - pow(beta1, iter));
        sdw[w_elem_idx] /= (1.0f - pow(beta2, iter));

        w[w_elem_idx] -= (lr * (vdw[w_elem_idx] / (sqrt(sdw[w_elem_idx]) + EPSILON)) / batch_size);
    }
}

__global__ void k_adam_bias_step(float *b, float *db, float *vdb, float *sdb, int b_cnt, float lr, float beta1, float beta2, int batch_size, int iter)
{
    int b_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_elem_idx < b_cnt)
    {
        vdb[b_elem_idx] = beta1 * vdb[b_elem_idx] + (1.0f - beta1) * db[b_elem_idx];
        sdb[b_elem_idx] = beta2 * sdb[b_elem_idx] + (1.0f - beta2) * (db[b_elem_idx] * db[b_elem_idx]);

        vdb[b_elem_idx] /= (1.0f - pow(beta1, iter));
        sdb[b_elem_idx] /= (1.0f - pow(beta2, iter));

        b[b_elem_idx] -= (lr * (vdb[b_elem_idx] / (sqrt(sdb[b_elem_idx])) + EPSILON) / batch_size);
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

void SGD::step(int batch_size, int iter)
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

SGDMomentum::SGDMomentum(std::vector<Parameters *> model_params, float learning_rate, float beta1)
    : Optimizer(model_params, learning_rate)
{
    this->beta1_ = beta1;

    for (Parameters *params : model_params)
    {
        this->vdws_.push_back(NdArray::zeros(true, params->weight_gradients()->shape()));
        this->vdbs_.push_back(NdArray::zeros(true, params->bias_gradients()->shape()));
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

void SGDMomentum::step(int batch_size, int iter)
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
                                                                                                   w_cnt, this->lr_, this->beta1_, batch_size, iter);
        k_sgd_momentum_bias_step<<<b_cnt / CUDA_THREADS_PER_BLOCK + 1, CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), vdb->data(),
                                                                                                 b_cnt, this->lr_, this->beta1_, batch_size, iter);

        params->zero_grad();
    }
}

Adam::Adam(std::vector<Parameters *> model_params, float learning_rate, float beta1, float beta2)
    : Optimizer(model_params, learning_rate)
{
    this->beta1_ = beta1;
    this->beta2_ = beta2;

    for (Parameters *params : model_params)
    {
        this->vdws_.push_back(NdArray::zeros(true, params->weight_gradients()->shape()));
        this->vdbs_.push_back(NdArray::zeros(true, params->bias_gradients()->shape()));
        this->sdws_.push_back(NdArray::zeros(true, params->weight_gradients()->shape()));
        this->sdbs_.push_back(NdArray::zeros(true, params->bias_gradients()->shape()));
    }
}

Adam::~Adam()
{
    for (int i = 0; i < this->vdws_.size(); i++)
    {
        delete this->vdws_[i];
        delete this->vdbs_[i];
        delete this->sdws_[i];
        delete this->sdbs_[i];
    }
}

void Adam::step(int batch_size, int iter)
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
        NdArray *sdw = this->sdws_[i];
        NdArray *sdb = this->sdbs_[i];

        int w_cnt = w->count();
        int b_cnt = b->count();

        k_adam_weight_step<<<w_cnt / CUDA_THREADS_PER_BLOCK + 1, CUDA_THREADS_PER_BLOCK>>>(w->data(), dw->data(), vdw->data(), sdw->data(),
                                                                                                   w_cnt, this->lr_, this->beta1_, this->beta2_, batch_size, iter);
        k_adam_bias_step<<<b_cnt / CUDA_THREADS_PER_BLOCK + 1, CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), vdb->data(), sdb->data(),
                                                                                                 b_cnt, this->lr_, this->beta1_, this->beta2_, batch_size, iter);

        params->zero_grad();
    }
}