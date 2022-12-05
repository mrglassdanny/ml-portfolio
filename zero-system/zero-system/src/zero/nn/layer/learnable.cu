#include "learnable.cuh"

using namespace zero::core;
using namespace zero::nn::layer;

Parameters::Parameters(Shape w_shape, Shape b_shape, int fan_in, int fan_out)
{
    this->w_ = Tensor::random(true, w_shape, 0.0f, sqrt(1.0f / fan_in));
    this->b_ = Tensor::zeros(true, b_shape);
    this->dw_ = Tensor::zeros(true, w_shape);
    this->db_ = Tensor::zeros(true, b_shape);
}

Parameters::~Parameters()
{
    delete this->w_;
    delete this->b_;
    delete this->dw_;
    delete this->db_;
}

void Parameters::zero_grad()
{
    this->dw_->zeros();
    this->db_->zeros();
}

size_t Parameters::count()
{
    return this->w_->count() + this->b_->count();
}

Tensor *Parameters::weights()
{
    return this->w_;
}

Tensor *Parameters::biases()
{
    return this->b_;
}

Tensor *Parameters::weight_gradients()
{
    return this->dw_;
}

Tensor *Parameters::bias_gradients()
{
    return this->db_;
}

Learnable::Learnable(bool shared_params)
{
    this->params_ = nullptr;
    this->shared_params_ = shared_params;
}

Learnable::~Learnable()
{
    if (!this->shared_params_)
    {
        delete params_;
    }
}

Parameters *Learnable::parameters()
{
    return this->params_;
}

void Learnable::share_parameters(Parameters *params)
{
    if (this->params_ != nullptr)
    {
        delete this->params_;
    }

    this->params_ = params;
    this->shared_params_ = true;
}

void Learnable::load_parameters()
{
    this->shared_params_ = false;
}