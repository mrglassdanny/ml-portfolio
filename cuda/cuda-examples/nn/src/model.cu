#include "model.cuh"

Model::Model(Loss *loss)
{
    this->loss_ = loss;
}

Model::~Model()
{
    for (Layer *lyr : this->lyrs_)
    {
        delete lyr;
    }

    delete this->loss_;
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

void Model::backward(NdArray *p, NdArray *y)
{
    y->to_cuda();

    NdArray *dl = this->loss_->derive(p, y);

    int lst_lyr_idx = this->lyrs_.size() - 1;
    for (int i = lst_lyr_idx; i >= 0; i--)
    {
        Layer *lyr = this->lyrs_[i];
        dl = lyr->backward(dl);
    }

    delete dl;
}

float Model::loss(NdArray *p, NdArray *y)
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