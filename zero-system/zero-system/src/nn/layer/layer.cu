#include "layer.cuh"

using namespace nn::layer;

Layer::~Layer()
{
    delete this->n_;
}

void Layer::summarize()
{
    std::string cls_name(typeid(*this).name());
    for (int i = cls_name.size(); i < 26; i++)
    {
        cls_name.push_back(' ');
    }

    printf("%s\t", cls_name.c_str());
    this->input_shape().print();
    printf(" -> ");
    this->output_shape().print();
}

int Layer::batch_size()
{
    return this->n_->shape()[0];
}

NdArray *Layer::neurons()
{
    return this->n_;
}

void Layer::copy_neurons(NdArray *n)
{
    this->n_->copy(n);
}
