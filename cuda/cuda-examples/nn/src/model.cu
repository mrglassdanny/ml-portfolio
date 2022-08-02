#include "model.cuh"

using namespace nn;

Model::Model()
{
    this->loss_ = nullptr;
    this->optim_ = nullptr;
}

Model::~Model()
{
    for (Layer *lyr : this->lyrs_)
    {
        delete lyr;
    }

    if (this->loss_ != nullptr)
    {
        delete this->loss_;
    }

    if (this->optim_ != nullptr)
    {
        delete this->optim_;
    }
}

Layer *Model::first_layer()
{
    return this->lyrs_[0];
}

Layer *Model::last_layer()
{
    return this->lyrs_[this->lyrs_.size() - 1];
}

int Model::batch_size()
{
    return this->first_layer()->batch_size();
}

void Model::lock_batch_size(int batch_size)
{
    if (this->last_layer()->batch_size() == batch_size)
    {
        return;
    }

    for (Layer *lyr : this->lyrs_)
    {
        lyr->lock_batch_size(batch_size);
    }
}

void Model::add_layer(Layer *lyr)
{
    this->lyrs_.push_back(lyr);
}

void Model::set_loss(Loss *loss)
{
    this->loss_ = loss;
}

void Model::set_optimizer(Optimizer *optim)
{
    this->optim_ = optim;
}

std::vector<Layer *> Model::layers()
{
    return this->lyrs_;
}

std::vector<Parameters *> Model::parameters()
{
    std::vector<Parameters *> params;

    for (Layer *lyr : this->lyrs_)
    {
        if (Learnable *lrn = dynamic_cast<Learnable *>(lyr))
        {
            params.push_back(lrn->parameters());
        }
    }

    return params;
}

NdArray *Model::forward(NdArray *x)
{
    if (this->lyrs_.size() == 0)
    {
        return nullptr;
    }

    x->to_cuda();

    int batch_size = x->shape().dim(0);
    this->lock_batch_size(batch_size);

    this->first_layer()->copy_neurons(x);

    for (int i = 0; i < this->lyrs_.size() - 1; i++)
    {
        Layer *lyr = this->lyrs_[i];
        Layer *nxt_lyr = this->lyrs_[i + 1];

        lyr->evaluate(nxt_lyr->neurons());
    }

    Layer *lst_lyr = this->last_layer();

    NdArray *p = new NdArray(true, lst_lyr->neurons()->shape());
    lst_lyr->evaluate(p);

    return p;
}

NdArray *Model::loss(NdArray *p, NdArray *y)
{
    if (this->loss_ == nullptr)
    {
        return nullptr;
    }

    p->to_cuda();
    y->to_cuda();

    NdArray *out = NdArray::zeros(true, p->shape());

    this->loss_->evaluate(p, y, out);

    return out;
}

void Model::backward(NdArray *p, NdArray *y)
{
    if (this->loss_ == nullptr)
    {
        return;
    }

    p->to_cuda();
    y->to_cuda();

    NdArray *dl = this->loss_->derive(p, y);

    int lst_lyr_idx = this->lyrs_.size() - 1;
    for (int i = lst_lyr_idx; i >= 0; i--)
    {
        Layer *lyr = this->lyrs_[i];
        dl = lyr->derive(dl);
    }

    delete dl;
}

void Model::step()
{
    if (this->optim_ == nullptr)
    {
        return;
    }

    this->optim_->step(this->batch_size());
}

