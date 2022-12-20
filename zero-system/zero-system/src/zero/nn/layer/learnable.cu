#include "learnable.cuh"

using namespace zero::core;
using namespace zero::nn::layer;

void Initializer::summarize()
{
    std::string cls_name(typeid(*this).name());
    printf("%s", cls_name.c_str());
}

void Xavier::initialize(Tensor *tensor, int fan_in, int fan_out)
{
    tensor->random(0.0f, sqrt(1.0f / fan_in));
}

Initializer *Xavier::copy()
{
    return new Xavier();
}

void He::initialize(Tensor *tensor, int fan_in, int fan_out)
{
    tensor->random(0.0f, sqrt(2.0f / fan_in));
}

Initializer *He::copy()
{
    return new He();
}

Parameters::Parameters(Shape w_shape, Shape b_shape, int fan_in, int fan_out, Initializer *initializer)
{
    this->w_ = Tensor::zeros(true, w_shape);
    this->b_ = Tensor::zeros(true, b_shape);
    this->dw_ = Tensor::zeros(true, w_shape);
    this->db_ = Tensor::zeros(true, b_shape);

    if (initializer != nullptr)
    {
        initializer->initialize(this->w_, fan_in, fan_out);
    }
}

Parameters::~Parameters()
{
    delete this->w_;
    delete this->b_;
    delete this->dw_;
    delete this->db_;
}

void Parameters::save(FILE *file)
{
    fwrite(this->w_, this->w_->size(), 1, file);
    fwrite(this->b_, this->b_->size(), 1, file);
}

void Parameters::load(FILE *file)
{
    fread(this->w_, this->w_->size(), 1, file);
    fread(this->b_, this->b_->size(), 1, file);
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