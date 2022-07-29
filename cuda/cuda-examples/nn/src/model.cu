#include "model.cuh"

Model::Model()
{
}

Model::~Model()
{
    for (Layer *lyr : this->lyrs_)
    {
        delete lyr;
    }
}

void Model::add_layer(Layer *lyr)
{
    this->lyrs_.push_back(lyr);
}

void Model::linear(int in_cnt, int out_cnt)
{
    this->add_layer(new Linear(in_cnt, out_cnt));
}

void Model::sigmoid(int in_cnt)
{
    this->add_layer(new Sigmoid(in_cnt));
}

void Model::lock_batch_size(int batch_size)
{
    if (this->lyrs_[0]->neurons()->batch_size() == batch_size)
    {
        return;
    }

    for (Layer *lyr : this->lyrs_)
    {
        lyr->neurons()->change_dim(0, batch_size);
    }
}

NdArray *Model::forward(NdArray *x)
{
    x->to_cuda();

    int batch_size = x->shape().dim(0);

    int lst_lyr_idx = this->lyrs_.size() - 1;

    for (int i = 0; i < lst_lyr_idx; i++)
    {
        Layer *lyr = this->lyrs_[i];
        Layer *nxt_lyr = this->lyrs_[i + 1];

        lyr->forward(nxt_lyr->neurons());
    }

    Layer *lst_lyr = this->lyrs_[lst_lyr_idx];

    NdArray *p = new NdArray(true, lst_lyr->neurons()->shape());
    lst_lyr->forward(p);

    return p;
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