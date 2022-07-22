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

ArrayNd *Model::forward(ArrayNd *x)
{
    x->to_cuda();

    int lst_lyr_idx = this->lyrs_.size() - 1;

    for (int i = 0; i < lst_lyr_idx; i++)
    {
        Layer *lyr = this->lyrs_[i];
        Layer *nxt_lyr = this->lyrs_[i + 1];

        lyr->forward(nxt_lyr->n());
    }

    Layer *lst_lyr = this->lyrs_[lst_lyr_idx];

    return NULL;
}

MSELoss::MSELoss()
{
}

MSELoss::~MSELoss()
{
}

float MSELoss::evaluate(ArrayNd *p, ArrayNd *y)
{
}

ArrayNd *MSELoss::derive(ArrayNd *p, ArrayNd *y)
{
    return NULL;
}