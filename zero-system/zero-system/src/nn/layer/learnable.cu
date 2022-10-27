#include "learnable.cuh"

using namespace nn::layer;

Parameters::Parameters(Shape w_shape, Shape b_shape, int fan_in, int fan_out)
{
    this->w_ = NdArray::random(true, w_shape, 0.0f, sqrt(1.0f / fan_in));
    // this->w_ = NdArray::ones(true, w_shape);
    this->b_ = NdArray::zeros(true, b_shape);
    this->dw_ = NdArray::zeros(true, w_shape);
    this->db_ = NdArray::zeros(true, b_shape);
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

NdArray *Parameters::weights()
{
    return this->w_;
}

NdArray *Parameters::biases()
{
    return this->b_;
}

NdArray *Parameters::weight_gradients()
{
    return this->dw_;
}

NdArray *Parameters::bias_gradients()
{
    return this->db_;
}

Learnable::~Learnable()
{
    delete params_;
}

Parameters *Learnable::parameters()
{
    return this->params_;
}
