#include "optim.cuh"

using namespace zero::core;
using namespace zero::nn::layer;
using namespace zero::nn::optim;

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

__global__ void k_sgd_momentum_weight_step(float *w, float *dw, float *mdw, int w_cnt, float lr, float beta1, int step_num, int batch_size)
{
    int w_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (w_elem_idx < w_cnt)
    {
        mdw[w_elem_idx] = beta1 * mdw[w_elem_idx] + (1.0f - beta1) * dw[w_elem_idx];

        float corrected_mdw = mdw[w_elem_idx] / (1.0f - pow(beta1, step_num));

        w[w_elem_idx] -= (lr * corrected_mdw / batch_size);
    }
}

__global__ void k_sgd_momentum_bias_step(float *b, float *db, float *mdb, int b_cnt, float lr, float beta1, int step_num, int batch_size)
{
    int b_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_elem_idx < b_cnt)
    {
        mdb[b_elem_idx] = beta1 * mdb[b_elem_idx] + (1.0f - beta1) * db[b_elem_idx];

        float corrected_mdb = mdb[b_elem_idx] / (1.0f - pow(beta1, step_num));

        b[b_elem_idx] -= (lr * corrected_mdb / batch_size);
    }
}

__global__ void k_adam_weight_step(float *w, float *dw, float *mdw, float *vdw, int w_cnt, float lr, float beta1, float beta2, int step_num, int batch_size)
{
    int w_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (w_elem_idx < w_cnt)
    {
        mdw[w_elem_idx] = beta1 * mdw[w_elem_idx] + (1.0f - beta1) * dw[w_elem_idx];
        vdw[w_elem_idx] = beta2 * vdw[w_elem_idx] + (1.0f - beta2) * (dw[w_elem_idx] * dw[w_elem_idx]);

        float corrected_mdw = mdw[w_elem_idx] / (1.0f - pow(beta1, step_num));
        float corrected_vdw = vdw[w_elem_idx] / (1.0f - pow(beta2, step_num));

        w[w_elem_idx] -= (lr * (corrected_mdw / (sqrt(corrected_vdw) + EPSILON)) / batch_size);
    }
}

__global__ void k_adam_bias_step(float *b, float *db, float *mdb, float *vdb, int b_cnt, float lr, float beta1, float beta2, int step_num, int batch_size)
{
    int b_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_elem_idx < b_cnt)
    {
        mdb[b_elem_idx] = beta1 * mdb[b_elem_idx] + (1.0f - beta1) * db[b_elem_idx];
        vdb[b_elem_idx] = beta2 * vdb[b_elem_idx] + (1.0f - beta2) * (db[b_elem_idx] * db[b_elem_idx]);

        float corrected_mdb = mdb[b_elem_idx] / (1.0f - pow(beta1, step_num));
        float corrected_vdb = vdb[b_elem_idx] / (1.0f - pow(beta2, step_num));

        b[b_elem_idx] -= (lr * (corrected_mdb / (sqrt(corrected_vdb) + EPSILON)) / batch_size);
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

void Optimizer::set_learning_rate(float learning_rate)
{
    this->lr_ = learning_rate;
}

void Optimizer::scale_learning_rate(float factor)
{
    this->lr_ *= factor;
}

SGD::SGD(std::vector<Parameters *> model_params, float learning_rate)
    : Optimizer(model_params, learning_rate)
{
}

void SGD::step(int batch_size)
{
    for (Parameters *params : this->model_params_)
    {
        Tensor *w = params->weights();
        Tensor *b = params->biases();
        Tensor *dw = params->weight_gradients();
        Tensor *db = params->bias_gradients();

        int w_cnt = w->count();
        int b_cnt = b->count();

        k_sgd_weight_step<<<w_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(w->data(), dw->data(), w_cnt, this->lr_, batch_size);
        k_sgd_bias_step<<<b_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), b_cnt, this->lr_, batch_size);

        params->zero_grad();
    }

    this->step_num_++;
}

SGDMomentum::SGDMomentum(std::vector<Parameters *> model_params, float learning_rate, float beta1)
    : Optimizer(model_params, learning_rate)
{
    this->beta1_ = beta1;

    for (Parameters *params : model_params)
    {
        this->mdws_.push_back(Tensor::zeros(true, params->weight_gradients()->shape()));
        this->mdbs_.push_back(Tensor::zeros(true, params->bias_gradients()->shape()));
    }
}

SGDMomentum::~SGDMomentum()
{
    for (int i = 0; i < this->mdws_.size(); i++)
    {
        delete this->mdws_[i];
        delete this->mdbs_[i];
    }
}

void SGDMomentum::step(int batch_size)
{
    for (int i = 0; i < this->model_params_.size(); i++)
    {
        Parameters *params = this->model_params_[i];

        Tensor *w = params->weights();
        Tensor *b = params->biases();
        Tensor *dw = params->weight_gradients();
        Tensor *db = params->bias_gradients();
        Tensor *mdw = this->mdws_[i];
        Tensor *mdb = this->mdbs_[i];

        int w_cnt = w->count();
        int b_cnt = b->count();

        k_sgd_momentum_weight_step<<<w_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(w->data(), dw->data(), mdw->data(),
                                                                                                                       w_cnt, this->lr_, this->beta1_, this->step_num_, batch_size);
        k_sgd_momentum_bias_step<<<b_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), mdb->data(),
                                                                                                                     b_cnt, this->lr_, this->beta1_, this->step_num_, batch_size);

        params->zero_grad();
    }

    this->step_num_++;
}

Adam::Adam(std::vector<Parameters *> model_params, float learning_rate, float beta1, float beta2)
    : Optimizer(model_params, learning_rate)
{
    this->beta1_ = beta1;
    this->beta2_ = beta2;

    for (Parameters *params : model_params)
    {
        this->mdws_.push_back(Tensor::zeros(true, params->weight_gradients()->shape()));
        this->mdbs_.push_back(Tensor::zeros(true, params->bias_gradients()->shape()));
        this->vdws_.push_back(Tensor::zeros(true, params->weight_gradients()->shape()));
        this->vdbs_.push_back(Tensor::zeros(true, params->bias_gradients()->shape()));
    }
}

Adam::~Adam()
{
    for (int i = 0; i < this->mdws_.size(); i++)
    {
        delete this->mdws_[i];
        delete this->mdbs_[i];
        delete this->vdws_[i];
        delete this->vdbs_[i];
    }
}

void Adam::step(int batch_size)
{
    for (int i = 0; i < this->model_params_.size(); i++)
    {
        Parameters *params = this->model_params_[i];

        Tensor *w = params->weights();
        Tensor *b = params->biases();
        Tensor *dw = params->weight_gradients();
        Tensor *db = params->bias_gradients();
        Tensor *mdw = this->mdws_[i];
        Tensor *mdb = this->mdbs_[i];
        Tensor *vdw = this->vdws_[i];
        Tensor *vdb = this->vdbs_[i];

        int w_cnt = w->count();
        int b_cnt = b->count();

        k_adam_weight_step<<<w_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(w->data(), dw->data(), mdw->data(), vdw->data(),
                                                                                                               w_cnt, this->lr_, this->beta1_, this->beta2_, this->step_num_, batch_size);
        k_adam_bias_step<<<b_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), mdb->data(), vdb->data(),
                                                                                                             b_cnt, this->lr_, this->beta1_, this->beta2_, this->step_num_, batch_size);

        params->zero_grad();
    }

    this->step_num_++;
}