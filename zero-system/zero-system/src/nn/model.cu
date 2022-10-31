#include "model.cuh"

using namespace nn;

Model::Model()
{
    this->loss_ = nullptr;
    this->optim_ = nullptr;

    this->validations_ = Validations{false, false, false};
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

NdArray *Model::forward(NdArray *x)
{
    this->validate_layers();
    this->validate_input(x);

    x->to_cuda();

    this->first_layer()->copy_neurons(x);

    for (int i = 1; i < this->lyrs_.size(); i++)
    {
        this->lyrs_[i]->neurons()->zeros();
    }

    NdArray *p = NdArray::zeros(true, this->output_shape());

    for (int lyr_idx = 0; lyr_idx < this->lyrs_.size() - 1; lyr_idx++)
    {
        Layer *lyr = this->lyrs_[lyr_idx];
        Layer *nxt_lyr = this->lyrs_[lyr_idx + 1];

        lyr->evaluate(nxt_lyr->neurons());

        if (FullResidual *fr_lyr = dynamic_cast<FullResidual *>(lyr))
        {
            int residual_param_idx = 0;
            for (int nxt_lyr_idx = lyr_idx + 2; nxt_lyr_idx < this->lyrs_.size(); nxt_lyr_idx++, residual_param_idx++)
            {
                nxt_lyr = this->lyrs_[nxt_lyr_idx];
                fr_lyr->evaluate_residual(nxt_lyr->neurons(), residual_param_idx);
            }
            fr_lyr->evaluate_residual(p, residual_param_idx);
        }
    }

    Layer *lst_lyr = this->last_layer();
    lst_lyr->evaluate(p);

    return p;
}

float Model::loss(NdArray *p, NdArray *y)
{
    this->validate_loss();
    this->validate_output(y);

    p->to_cuda();
    y->to_cuda();

    NdArray *losses = NdArray::zeros(true, p->shape());
    this->loss_->evaluate(p, y, losses);
    float mean_loss = losses->sum() / this->batch_size();

    delete losses;
    return mean_loss;
}

float Model::accuracy(NdArray *p, NdArray *y)
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

void Model::backward(NdArray *p, NdArray *y)
{
    this->validate_layers();
    this->validate_loss();
    this->validate_output(y);

    p->to_cuda();
    y->to_cuda();

    NdArray *loss_gradients = this->loss_->derive(p, y);
    NdArray *prev_n = p;

    for (int lyr_idx = this->lyrs_.size() - 1; lyr_idx >= 0; lyr_idx--)
    {
        Layer *lyr = this->lyrs_[lyr_idx];

        lyr->derive(loss_gradients, prev_n);

        for (int nxt_lyr_idx = lyr_idx - 1, residual_param_idx = 0; nxt_lyr_idx >= 0; nxt_lyr_idx--, residual_param_idx++)
        {
            if (FullResidual *fr_lyr = dynamic_cast<FullResidual *>(this->lyrs_[nxt_lyr_idx]))
            {
                fr_lyr->derive_residual(loss_gradients, prev_n, residual_param_idx);
            }
        }

        if (lyr_idx == this->lyrs_.size() - 1)
        {
            delete loss_gradients;
        }

        loss_gradients = lyr->neuron_gradients();
        prev_n = lyr->neurons();
    }
}

void Model::step()
{
    this->validate_optimizer();

    this->optim_->step(this->batch_size());

    for (Layer *lyr : this->lyrs_)
    {
        lyr->zero_grad();
    }
}

Shape Model::input_shape()
{
    return this->first_layer()->input_shape();
}

Shape Model::output_shape()
{
    return this->last_layer()->output_shape();
}

void Model::validate_layers()
{
    if (this->validations_.layers)
    {
        return;
    }

    if (this->lyrs_.size() == 0)
    {
        THROW_ERROR("MODEL VALIDATION FAILED: no layers");
    }

    for (int i = 0; i < this->lyrs_.size() - 1; i++)
    {
        Layer *lyr = this->lyrs_[i];
        Layer *nxt_lyr = this->lyrs_[i + 1];

        if (lyr->output_shape() != nxt_lyr->input_shape())
        {
            THROW_ERROR("MODEL VALIDATION FAILED: layer output shape does not match next layer input shape");
        }
    }

    for (Layer *lyr : this->lyrs_)
    {
        lyr->validate();
    }

    if (this->output_shape().num_dims() != 2)
    {
        THROW_ERROR("MODEL VALIDATION FAILED: output shape must have 2 dimensions");
    }

    this->validations_.layers = true;
}

void Model::validate_loss()
{
    if (this->validations_.loss)
    {
        return;
    }

    if (this->loss_ == nullptr)
    {
        THROW_ERROR("MODEL LOSS VALIDATION FAILED: loss not set");
    }

    this->validations_.loss = true;
}

void Model::validate_optimizer()
{
    if (this->validations_.optimizer)
    {
        return;
    }

    if (this->optim_ == nullptr)
    {
        THROW_ERROR("MODEL OPTIMIZER VALIDATION FAILED: optimizer not set");
    }

    this->validations_.optimizer = true;
}

void Model::validate_input(NdArray *x)
{
    if (this->input_shape() != x->shape())
    {
        THROW_ERROR("MODEL INPUT VALIDATION FAILED: X shape does not match model input shape");
    }
}

void Model::validate_output(NdArray *y)
{
    if (this->output_shape() != y->shape())
    {
        THROW_ERROR("MODEL OUTPUT VALIDATION FAILED: Y shape does not match model output shape");
    }
}

void Model::validate_gradients(NdArray *x, NdArray *y, bool print_params)
{
    this->validate_layers();
    this->validate_loss();
    this->validate_input(x);
    this->validate_output(y);

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
            THROW_ERROR("MODEL GRADIENTS VALIDATION FAILED");
        }
    }
}

void Model::summarize()
{
    printf("=========================================================================== MODEL SUMMARY ===========================================================================\n");

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

void Model::add_layer(Layer *lyr)
{
    this->lyrs_.push_back(lyr);

    for (int i = 0; i < this->lyrs_.size() - 1; i++)
    {
        if (FullResidual *fr_lyr = dynamic_cast<FullResidual *>(this->lyrs_[i]))
        {
            fr_lyr->link(lyr);
        }
    }
}

void Model::set_loss(Loss *loss)
{
    this->loss_ = loss;
}

void Model::set_optimizer(Optimizer *optim)
{
    this->optim_ = optim;
}

void Model::linear(int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new Linear(this->output_shape(), Shape(this->batch_size(), out_feature_cnt), activation));
}

void Model::linear(Shape y_shape, ActivationType activation)
{
    this->add_layer(new Linear(this->output_shape(), y_shape, activation));
}

void Model::linear(int batch_size, int in_feature_cnt, int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new Linear(Shape(batch_size, in_feature_cnt), Shape(batch_size, out_feature_cnt), activation));
}

void Model::linear(Shape in_shape, int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new Linear(in_shape, Shape(in_shape[0], out_feature_cnt), activation));
}

void Model::conv2d(Shape filter_shape, ActivationType activation)
{
    this->add_layer(new Conv2d(this->output_shape(), filter_shape, Stride{1, 1}, activation));
}

void Model::conv2d(Shape filter_shape, Stride stride, ActivationType activation)
{
    this->add_layer(new Conv2d(this->output_shape(), filter_shape, stride, activation));
}

void Model::conv2d(Shape in_shape, Shape filter_shape, Stride stride, ActivationType activation)
{
    this->add_layer(new Conv2d(in_shape, filter_shape, stride, activation));
}

void Model::full_residual(int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new FullResidual(this->output_shape(), Shape(this->batch_size(), out_feature_cnt), activation));
}

void Model::full_residual(Shape y_shape, ActivationType activation)
{
    this->add_layer(new FullResidual(this->output_shape(), y_shape, activation));
}

void Model::full_residual(int batch_size, int in_feature_cnt, int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new FullResidual(Shape(batch_size, in_feature_cnt), Shape(batch_size, out_feature_cnt), activation));
}

void Model::full_residual(Shape in_shape, int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new FullResidual(in_shape, Shape(in_shape[0], out_feature_cnt), activation));
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
        if (Learnable *lrn_lyr = dynamic_cast<Learnable *>(lyr))
        {
            params.push_back(lrn_lyr->parameters());

            if (FullResidual *fr_lyr = dynamic_cast<FullResidual *>(lrn_lyr))
            {
                for (Parameters *r_params : fr_lyr->residual_parameters())
                {
                    params.push_back(r_params);
                }
            }
        }
    }

    return params;
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

void Model::change_batch_size(int batch_size)
{
    for (Layer *lyr : this->lyrs_)
    {
        lyr->change_batch_size(batch_size);
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
