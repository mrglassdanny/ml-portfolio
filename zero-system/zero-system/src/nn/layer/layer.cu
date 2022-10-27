#include "layer.cuh"

using namespace nn::layer;

Layer::~Layer()
{
    delete this->n_;
    delete this->dn_;
}

void Layer::reset_shape() {}

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

int Layer::batch_size()
{
    return this->n_->shape()[0];
}

void Layer::change_batch_size(int batch_size)
{
    this->n_->change_dim(0, batch_size);
    this->dn_->change_dim(0, batch_size);

    std::vector<int> default_n_dims = this->default_n_shape_.dims();
    default_n_dims[0] = batch_size;
    this->default_n_shape_ = Shape(default_n_dims);
}

NdArray *Layer::neurons()
{
    return this->n_;
}

NdArray *Layer::neuron_gradients()
{
    return this->dn_;
}

void Layer::copy_neurons(NdArray *n)
{
    this->n_->copy(n);
}

void Layer::zero_grad()
{
    this->dn_->zeros();
}
