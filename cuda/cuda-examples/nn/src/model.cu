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

    delete this->loss_;
}

ArrayNd *Model::forward(ArrayNd *x)
{
    x->to_cuda();

    int batch_size = x->dims().dim(0);

    this->lyrs_[0]->set_n(x);

    int lst_lyr_idx = this->lyrs_.size() - 1;

    for (int i = 0; i < lst_lyr_idx; i++)
    {
        Layer *lyr = this->lyrs_[i];
        Layer *nxt_lyr = this->lyrs_[i + 1];

        lyr->forward(nxt_lyr->n());
    }

    Layer *lst_lyr = this->lyrs_[lst_lyr_idx];

    ArrayNd *p = new ArrayNd(true, lst_lyr->n()->dims());
    lst_lyr->forward(p);

    return p;
}

void Model::backward(ArrayNd *p, ArrayNd *y)
{
    y->to_cuda();

    ArrayNd *dl = this->loss_->derive(p, y);

    int lst_lyr_idx = this->lyrs_.size() - 1;
    for (int i = lst_lyr_idx; i >= 0; i--)
    {
        Layer *lyr = this->lyrs_[i];
        dl = lyr->backward(dl);
    }

    delete dl;
}

float Model::loss(ArrayNd *p, ArrayNd *y)
{
    y->to_cuda();

    float loss_val = 0.0f;
    float *d_loss_val;

    cudaMalloc(&d_loss_val, sizeof(float));
    cudaMemset(d_loss_val, 0, sizeof(float));

    this->loss_->evaluate(p, y, d_loss_val);

    cudaMemcpy(&loss_val, d_loss_val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss_val);

    return loss_val;
}

void Model::step()
{
}