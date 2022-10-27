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

__global__ void k_enhanced_residual_inc_param_derivatives(float *in, float *in_n, float *n, float *dw, float *db,
                                                          int batch_size, int in_cnt, int n_cnt, int w_row_cnt, int w_col_cnt)
{
    int w_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int w_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (w_col_idx < w_col_cnt && w_row_idx < w_row_cnt)
    {
        int w_elem_idx = w_row_idx * w_col_cnt + w_col_idx;
        int n_idx = w_row_idx;
        int in_idx = w_col_idx;

        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
        {
            dw[w_elem_idx] += (in[batch_idx * in_cnt + in_idx] * n[batch_idx * n_cnt + n_idx]);

            if (w_row_idx == 0)
            {
                int b_elem_idx = w_col_idx;
                db[b_elem_idx] += in[batch_idx * in_cnt + in_idx];
            }
        }
    }
}

__global__ void k_enhanced_residual_agg_derivatives(float *in, float *w, float *out, int batch_size, int in_cnt, int w_row_cnt, int w_col_cnt, int out_cnt)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_idx < out_cnt && batch_idx < batch_size)
    {
        int out_elem_idx = batch_idx * w_row_cnt + out_idx;
        int w_row_idx = out_idx;

        for (int in_idx = 0; in_idx < in_cnt; in_idx++)
        {
            int w_col_idx = in_idx;
            out[out_elem_idx] += (in[batch_idx * in_cnt + in_idx] * w[w_row_idx * w_col_cnt + w_col_idx]);
        }
    }
}

EnhancedResidual::EnhancedResidual(Shape in_shape, Shape out_shape, ActivationType activation)
{
    this->n_ = new NdArray(true, in_shape);
    this->dn_ = new NdArray(true, in_shape);

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

void EnhancedResidual::derive_residual(NdArray *in, NdArray *in_n, int idx)
{
    NdArray *n = this->n_;
    NdArray *w = this->residual_params_[idx]->weights();
    NdArray *b = this->residual_params_[idx]->biases();
    NdArray *dw = this->residual_params_[idx]->weight_gradients();
    NdArray *db = this->residual_params_[idx]->bias_gradients();

    int w_row_cnt = w->shape()[0];
    int w_col_cnt = w->shape()[1];
    int out_cnt = w_col_cnt;

    {
        int grid_row_cnt = (w_row_cnt / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (w_col_cnt / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_enhanced_residual_inc_param_derivatives<<<grid_dims, block_dims>>>(in->data(), in_n->data(), n->data(), dw->data(), db->data(),
                                                                             this->batch_size(), out_cnt, this->in_features(),
                                                                             w_row_cnt, w_col_cnt);
    }

    {
        int grid_row_cnt = (this->batch_size() / CUDA_THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (this->in_features() / CUDA_THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);

        k_enhanced_residual_agg_derivatives<<<grid_dims, block_dims>>>(in->data(), w->data(), this->dn_->data(),
                                                                       this->batch_size(), out_cnt, w_row_cnt, w_col_cnt, this->in_features());
    }
}

void EnhancedResidual::compile(ERNN *ernn, int my_idx)
{
    int fan_in = this->input_shape().dims_size() / this->batch_size();

    std::vector<EnhancedResidual *> lyrs = ernn->layers();
    for (int i = my_idx + 1; i < lyrs.size(); i++)
    {
        EnhancedResidual *lyr = lyrs[i];
        int fan_out = lyr->output_shape().dims_size() / this->batch_size();

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

    for (int i = 1; i < this->lyrs_.size(); i++)
    {
        this->lyrs_[i]->neurons()->zeros();
    }
    NdArray *p = NdArray::zeros(true, this->output_shape());

    for (int i = 0; i < this->lyrs_.size() - 1; i++)
    {
        EnhancedResidual *lyr = this->lyrs_[i];
        Layer *nxt_lyr = this->lyrs_[i + 1];

        lyr->evaluate(nxt_lyr->neurons());

        int k = 0;
        for (int j = i + 2; j < this->lyrs_.size(); j++)
        {
            nxt_lyr = this->lyrs_[j];
            lyr->evaluate_residual(nxt_lyr->neurons(), k++);
        }
        lyr->evaluate_residual(p, k);
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

        int k = 0;
        for (int j = i - 1; j >= 0; j--)
        {
            this->lyrs_[j]->derive_residual(loss_gradients, prev_n, k++);
        }

        if (i == this->lyrs_.size() - 1)
        {
            delete loss_gradients;
        }

        loss_gradients = lyr->neuron_gradients();
        prev_n = lyr->neurons();
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

void ERNN::validate_gradients(NdArray *x, NdArray *y, bool print_params)
{
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

            w->set_val(i, w_val - EPSILON);
            p = this->forward(x);
            float left_loss = this->loss(p, y);
            delete p;

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

            b->set_val(i, b_val - EPSILON);
            p = this->forward(x);
            float left_loss = this->loss(p, y);
            delete p;

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

        if ((agg_grad_diff) / (agg_ana_grad + agg_num_grad) > EPSILON)
        {
            THROW_ERROR("ERNN GRADIENTS VALIDATION FAILED");
        }
    }
}

void ERNN::summarize()
{
    printf("=========================================================================== ERNN SUMMARY ============================================================================\n");

    printf("\nLayers: (%d)\n", this->lyrs_.size());
    for (int i = 0; i < this->lyrs_.size(); i++)
    {
        printf("\t%d\t", i + 1);
        this->lyrs_[i]->summarize();
        printf("\n");
    }
    printf("\n");

    printf("Loss: ");
    if (this->loss_ != nullptr)
    {
        this->loss_->summarize();
    }
    else
    {
        printf("None");
    }
    printf("\n\n");

    printf("Optimizer: ");
    if (this->optim_ != nullptr)
    {
        this->optim_->summarize();
    }
    else
    {
        printf("None");
    }
    printf("\n\n");

    printf("=====================================================================================================================================================================\n");
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

void ERNN::layer(int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new EnhancedResidual(this->output_shape(), Shape(this->batch_size(), out_feature_cnt), activation));
}

void ERNN::layer(Shape y_shape, ActivationType activation)
{
    this->add_layer(new EnhancedResidual(this->output_shape(), y_shape, activation));
}

void ERNN::layer(int batch_size, int in_feature_cnt, int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new EnhancedResidual(Shape(batch_size, in_feature_cnt), Shape(batch_size, out_feature_cnt), activation));
}

void ERNN::layer(Shape in_shape, int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new EnhancedResidual(in_shape, Shape(in_shape[0], out_feature_cnt), activation));
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
