#include "layer.cuh"

using namespace zero::core;
using namespace zero::nn::layer;

Layer::~Layer()
{
    delete this->n_;
    delete this->dn_;
}

void Layer::summarize()
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
}

int Layer::in_features()
{
    return this->input_shape().dims_size() / this->batch_size();
}

int Layer::out_features()
{
    return this->output_shape().dims_size() / this->batch_size();
}

int Layer::batch_size()
{
    return this->n_->shape()[0];
}

void Layer::change_batch_size(int batch_size)
{
    this->n_->change_dim(0, batch_size);
    this->dn_->change_dim(0, batch_size);
}

Tensor *Layer::neurons()
{
    return this->n_;
}

void Layer::copy_neurons(Tensor *n)
{
    this->n_->copy(n);
}

Tensor *Layer::neuron_gradients()
{
    return this->dn_;
}

void Layer::zero_grad()
{
    this->dn_->zeros();
}

ActivationType Layer::activation()
{
    return this->activation_;
}
