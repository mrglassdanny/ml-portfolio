#include "ernn.cuh"

using namespace nn;
using namespace nn::layer;

__global__ void k_enhanced_residual_evaluate(float *in, float *w, float *b, float *out,
                                             int batch_size, int in_cnt, int out_cnt)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx < out_cnt && batch_idx < batch_size)
    {
        int out_elem_idx = batch_idx * out_cnt + out_idx;
        int w_col_idx = out_idx;

        for (int in_idx = 0; in_idx < in_cnt; in_idx++)
        {
            int w_row_idx = in_idx;
            out[out_elem_idx] += (in[batch_idx * in_cnt + in_idx] * w[w_row_idx * out_cnt + w_col_idx]);
        }

        out[out_elem_idx] += b[w_col_idx];
    }
}

EnhancedResidual::EnhancedResidual(Shape in_shape, Shape out_shape, ActivationType activation)
{
    this->n_ = new NdArray(true, in_shape);
    this->default_n_shape_ = in_shape;

    int in_cnt = (in_shape.dims_size() / this->batch_size());
    int out_cnt = (out_shape.dims_size() / this->batch_size());

    this->params_ = new Parameters(Shape(in_cnt, out_cnt), Shape(out_cnt), in_cnt, out_cnt);

    this->activation_ = activation;
}

EnhancedResidual::~EnhancedResidual()
{
    for (Parameters *rp : this->residual_params_)
    {
        delete rp;
    }
}

void EnhancedResidual::evaluate_residual(NdArray *out, int idx)
{
    int out_feature_cnt = out->dims_size() / this->batch_size();

    int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (out_feature_cnt / CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

    NdArray *n = this->n_;
    NdArray *w = this->residual_params_[idx]->weights();
    NdArray *b = this->residual_params_[idx]->biases();

    k_enhanced_residual_evaluate<<<grid_dims, block_dims>>>(n->data(), w->data(), b->data(), out->data(),
                                                            this->batch_size(), this->in_features(), out_feature_cnt);
}

void EnhancedResidual::compile(ERNN *ernn, int my_idx)
{
    int fan_in = this->input_shape().dims_size() / this->batch_size();

    std::vector<EnhancedResidual *> lyrs = ernn->layers();
    for (int i = my_idx; i < lyrs.size(); i++)
    {
        EnhancedResidual *lyr = lyrs[i];
        int fan_out = lyr->input_shape().dims_size() / this->batch_size();

        this->residual_params_.push_back(new Parameters(Shape(fan_in, fan_out), Shape(fan_out), fan_in, fan_out));
    }
}

std::vector<Parameters *> EnhancedResidual::residual_parameters()
{
    return this->residual_params_;
}

ERNN::ERNN()
{
    this->loss_ = nullptr;
    this->optim_ = nullptr;
}

ERNN::~ERNN()
{
    for (EnhancedResidual *lyr : this->lyrs_)
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

NdArray *ERNN::forward(NdArray *x)
{
    x->to_cuda();

    this->first_layer()->copy_neurons(x);

    for (int i = 0; i < this->lyrs_.size() - 1; i++)
    {
        Layer *nxt_lyr = this->lyrs_[i + 1];
        nxt_lyr->neurons()->zeros();
    }
    NdArray *p = NdArray::zeros(true, this->output_shape());

    for (int i = 0; i < this->lyrs_.size() - 1; i++)
    {
        EnhancedResidual *lyr = this->lyrs_[i];
        Layer *nxt_lyr = this->lyrs_[i + 1];

        lyr->evaluate(nxt_lyr->neurons());

        int k = 0;
        for (int j = i + 2; j < this->lyrs_.size() - 1; j++)
        {
            nxt_lyr = this->lyrs_[j];
            lyr->evaluate_residual(nxt_lyr->neurons(), k++);
        }

        lyr->evaluate_residual(p, k + 1);
    }

    Layer *lst_lyr = this->last_layer();
    lst_lyr->evaluate(p);

    return p;
}

float ERNN::loss(NdArray *p, NdArray *y)
{
    p->to_cuda();
    y->to_cuda();

    NdArray *losses = NdArray::zeros(true, p->shape());
    this->loss_->evaluate(p, y, losses);
    float mean_loss = losses->sum() / this->batch_size();

    delete losses;
    return mean_loss;
}

float ERNN::accuracy(NdArray *p, NdArray *y)
{
    int correct_cnt = 0;

    int batch_size = this->batch_size();
    int output_cnt = p->shape()[1];

    if (output_cnt > 1)
    {
        for (int i = 0; i < batch_size; i++)
        {
            float max_val = p->get_val(i * output_cnt + 0);
            int max_idx = 0;
            for (int j = 1; j < output_cnt; j++)
            {
                float val = p->get_val(i * output_cnt + j);
                if (val > max_val)
                {
                    max_val = val;
                    max_idx = j;
                }
            }

            if (y->get_val(i * output_cnt + max_idx) == 1.0f)
            {
                correct_cnt++;
            }
        }
    }
    else
    {
        // Assume value is between 0 and 1.

        for (int i = 0; i < batch_size; i++)
        {
            float p_val = p->get_val(i) >= 0.50 ? 1.0f : 0.0f;

            if (p_val == y->get_val(i))
            {
                correct_cnt++;
            }
        }
    }

    return ((float)correct_cnt / (float)batch_size);
}

void ERNN::backward(NdArray *p, NdArray *y)
{
    p->to_cuda();
    y->to_cuda();

    NdArray *loss_gradients = this->loss_->derive(p, y);
    NdArray *prev_n = p;

    for (int i = this->lyrs_.size() - 1; i >= 0; i--)
    {
        EnhancedResidual *lyr = this->lyrs_[i];

        lyr->derive(loss_gradients, prev_n);

        if (i == this->lyrs_.size() - 1)
        {
            delete loss_gradients;
        }

        loss_gradients = lyr->neuron_gradients();
        prev_n = this->lyrs_[i]->neurons();
    }
}

void ERNN::step()
{
    this->optim_->step(this->batch_size());
}

Shape ERNN::input_shape()
{
    return this->first_layer()->input_shape();
}

Shape ERNN::output_shape()
{
    return this->last_layer()->output_shape();
}

void ERNN::add_layer(EnhancedResidual *lyr)
{
    this->lyrs_.push_back(lyr);
}

void ERNN::set_loss(Loss *loss)
{
    this->loss_ = loss;
}

void ERNN::set_optimizer(Optimizer *optim)
{
    this->optim_ = optim;
}

std::vector<EnhancedResidual *> ERNN::layers()
{
    return this->lyrs_;
}

std::vector<Parameters *> ERNN::parameters()
{
    std::vector<Parameters *> params;

    for (EnhancedResidual *lyr : this->lyrs_)
    {
        params.push_back(lyr->parameters());

        for (Parameters *rp : lyr->residual_parameters())
        {
            params.push_back(rp);
        }
    }

    return params;
}

EnhancedResidual *ERNN::first_layer()
{
    return this->lyrs_[0];
}

EnhancedResidual *ERNN::last_layer()
{
    return this->lyrs_[this->lyrs_.size() - 1];
}

int ERNN::batch_size()
{
    return this->first_layer()->batch_size();
}

void ERNN::compile()
{
    for (int i = 0; i < this->lyrs_.size(); i++)
    {
        this->lyrs_[i]->compile(this, i);
    }
}
