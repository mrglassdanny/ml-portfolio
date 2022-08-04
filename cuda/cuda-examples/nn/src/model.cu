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

    int batch_size = x->shape()[0];
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

float Model::loss(NdArray *p, NdArray *y)
{
    if (this->loss_ == nullptr)
    {
        return 0.0f;
    }

    p->to_cuda();
    y->to_cuda();

    NdArray *losses = NdArray::zeros(true, p->shape());

    this->loss_->evaluate(p, y, losses);

    float mean_losses = losses->mean();

    delete losses;

    return mean_losses;
}

void Model::backward(NdArray *p, NdArray *y)
{
    if (this->loss_ == nullptr)
    {
        return;
    }

    p->to_cuda();
    y->to_cuda();

    NdArray *loss_gradients = this->loss_->derive(p, y);

    for (int i = this->lyrs_.size() - 1; i >= 0; i--)
    {
        loss_gradients = this->lyrs_[i]->derive(loss_gradients);
    }

    delete loss_gradients;
}

void Model::step()
{
    if (this->optim_ == nullptr)
    {
        return;
    }

    this->optim_->step(this->batch_size());
}

void Model::gradient_check(NdArray *x, NdArray *y, bool print_params)
{
    if (this->lyrs_.size() == 0)
    {
        printf("GRADIENT CHECK FAILED: layers not added");
        return;
    }

    if (this->loss_ == nullptr)
    {
        printf("GRADIENT CHECK FAILED: loss not set");
        return;
    }

    x->to_cuda();
    y->to_cuda();

    float agg_ana_grad = 0.0f;
    float agg_num_grad = 0.0f;
    float agg_grad_diff = 0.0f;

    NdArray *p = this->forward(x);
    this->backward(p, y);
    delete p;

    int param_idx = 0;
    for (Parameters *params : this->parameters())
    {
        NdArray *w = params->weights();
        NdArray *b = params->biases();
        NdArray *dw = params->weight_gradients();
        NdArray *db = params->bias_gradients();

        for (int i = 0; i < w->count(); i++)
        {
            float w_val = w->get_val(i);

            // Left:
            w->set_val(i, w_val - EPSILON);
            p = this->forward(x);
            float left_loss = this->loss(p, y);
            delete p;

            // Right:
            w->set_val(i, w_val + EPSILON);
            p = this->forward(x);
            float right_loss = this->loss(p, y);
            delete p;

            w->set_val(i, w_val);

            float num_grad = (right_loss - left_loss) / (2.0f * EPSILON);
            float ana_grad = dw->get_val(i);

            if (print_params)
            {
                printf("W: %d  %d\t|%f - %f| = %f\n", param_idx, i, ana_grad, num_grad, fabs(ana_grad - num_grad));
            }

            agg_ana_grad += (ana_grad * ana_grad);
            agg_num_grad += (num_grad * num_grad);
            agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));
        }

        for (int i = 0; i < b->count(); i++)
        {
            float b_val = b->get_val(i);

            // Left:
            b->set_val(i, b_val - EPSILON);
            p = this->forward(x);
            float left_loss = this->loss(p, y);
            delete p;

            // Right:
            b->set_val(i, b_val + EPSILON);
            p = this->forward(x);
            float right_loss = this->loss(p, y);
            delete p;

            b->set_val(i, b_val);

            float num_grad = (right_loss - left_loss) / (2.0f * EPSILON);
            float ana_grad = db->get_val(i);

            if (print_params)
            {
                printf("B: %d  %d\t|%f - %f| = %f\n", param_idx, i, ana_grad, num_grad, fabs(ana_grad - num_grad));
            }

            agg_ana_grad += (ana_grad * ana_grad);
            agg_num_grad += (num_grad * num_grad);
            agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));
        }

        param_idx++;
    }

    if ((agg_grad_diff) == 0.0f && (agg_ana_grad + agg_num_grad) == 0.0f)
    {
        printf("GRADIENT CHECK RESULT: %f\n", 0.0f);
    }
    else
    {
        printf("GRADIENT CHECK RESULT: %f\n", (agg_grad_diff) / (agg_ana_grad + agg_num_grad));
    }
}

void Model::performance_check(NdArray *x, NdArray *y, int epoch_cnt)
{
    CudaStopWatch *sw = new CudaStopWatch();

    sw->start();

    for (int i = 0; i < epoch_cnt; i++)
    {
        NdArray *p = this->forward(x);
        this->loss(p, y);
        this->backward(p, y);
        this->step();
        delete p;
    }

    sw->stop();

    sw->print_elapsed_seconds();

    delete sw;
}

