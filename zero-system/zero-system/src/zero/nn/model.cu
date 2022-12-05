#include "model.cuh"

using namespace zero::core;
using namespace zero::nn;

Model::Model()
{
    this->shared_params_ = false;

    this->loss_ = nullptr;
    this->optim_ = nullptr;

    this->validations_ = Validations{false, false, false};
}

Model::Model(bool shared_params)
{
    this->shared_params_ = shared_params;

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
        this->loss_ = nullptr;
    }

    if (this->optim_ != nullptr)
    {
        delete this->optim_;
        this->optim_ = nullptr;
    }
}

Model *Model::copy()
{
    auto model = new Model(true);

    for (auto lyr : this->lyrs_)
    {
        model->add_layer(lyr->copy());
    }

    model->loss_ = this->loss_;
    model->optim_ = this->optim_;

    model->validations_ = this->validations_;

    return model;
}

Tensor *Model::forward(Tensor *x)
{
    this->validate_layers();
    this->validate_input(x);

    x->to_cuda();

    this->first_layer()->copy_neurons(x);

    for (int i = 1; i < this->lyrs_.size(); i++)
    {
        this->lyrs_[i]->neurons()->zeros();
    }

    Tensor *p = Tensor::zeros(true, this->output_shape());

    for (int lyr_idx = 0; lyr_idx < this->lyrs_.size() - 1; lyr_idx++)
    {
        Layer *lyr = this->lyrs_[lyr_idx];
        Layer *nxt_lyr = this->lyrs_[lyr_idx + 1];

        lyr->evaluate(nxt_lyr->neurons());
    }

    Layer *lst_lyr = this->last_layer();
    lst_lyr->evaluate(p);

    return p;
}

float Model::loss(Tensor *p, Tensor *y)
{
    this->validate_loss();
    this->validate_output(y);

    p->to_cuda();
    y->to_cuda();

    Tensor *losses = Tensor::zeros(true, p->shape());
    this->loss_->evaluate(p, y, losses);
    float mean_loss = losses->sum() / this->batch_size();

    delete losses;
    return mean_loss;
}

float Model::accuracy(Tensor *p, Tensor *y)
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

void Model::backward(Tensor *p, Tensor *y)
{
    this->validate_layers();
    this->validate_loss();
    this->validate_output(y);

    p->to_cuda();
    y->to_cuda();

    Tensor *loss_gradients = this->loss_->derive(p, y);
    Tensor *prev_n = p;

    for (int lyr_idx = this->lyrs_.size() - 1; lyr_idx >= 0; lyr_idx--)
    {
        Layer *lyr = this->lyrs_[lyr_idx];

        lyr->derive(loss_gradients, prev_n);

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
        ZERO_CORE_THROW_ERROR("MODEL VALIDATION FAILED: no layers");
    }

    for (int i = 0; i < this->lyrs_.size() - 1; i++)
    {
        Layer *lyr = this->lyrs_[i];
        Layer *nxt_lyr = this->lyrs_[i + 1];

        if (lyr->output_shape() != nxt_lyr->input_shape())
        {
            ZERO_CORE_THROW_ERROR("MODEL VALIDATION FAILED: layer output shape does not match next layer input shape");
        }
    }

    for (Layer *lyr : this->lyrs_)
    {
        lyr->validate();
    }

    if (this->output_shape().num_dims() != 2)
    {
        ZERO_CORE_THROW_ERROR("MODEL VALIDATION FAILED: output shape must have 2 dimensions");
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
        ZERO_CORE_THROW_ERROR("MODEL LOSS VALIDATION FAILED: loss not set");
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
        ZERO_CORE_THROW_ERROR("MODEL OPTIMIZER VALIDATION FAILED: optimizer not set");
    }

    this->validations_.optimizer = true;
}

void Model::validate_input(Tensor *x)
{
    if (this->input_shape() != x->shape())
    {
        ZERO_CORE_THROW_ERROR("MODEL INPUT VALIDATION FAILED: X shape does not match model input shape");
    }
}

void Model::validate_output(Tensor *y)
{
    if (this->output_shape() != y->shape())
    {
        ZERO_CORE_THROW_ERROR("MODEL OUTPUT VALIDATION FAILED: Y shape does not match model output shape");
    }
}

void Model::validate_gradients(Tensor *x, Tensor *y, bool print_params)
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

    Tensor *p = this->forward(x);
    this->backward(p, y);
    delete p;

    int param_idx = 0;
    for (Parameters *params : this->parameters())
    {
        Tensor *w = params->weights();
        Tensor *b = params->biases();
        Tensor *dw = params->weight_gradients();
        Tensor *db = params->bias_gradients();

        for (int i = 0; i < w->count(); i++)
        {
            float w_val = w->get_val(i);

            w->set_val(i, w_val - ZERO_NN_EPSILON);
            p = this->forward(x);
            float left_loss = this->loss(p, y);
            delete p;

            w->set_val(i, w_val + ZERO_NN_EPSILON);
            p = this->forward(x);
            float right_loss = this->loss(p, y);
            delete p;

            w->set_val(i, w_val);

            float num_grad = (right_loss - left_loss) / (2.0f * ZERO_NN_EPSILON);
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

            b->set_val(i, b_val - ZERO_NN_EPSILON);
            p = this->forward(x);
            float left_loss = this->loss(p, y);
            delete p;

            b->set_val(i, b_val + ZERO_NN_EPSILON);
            p = this->forward(x);
            float right_loss = this->loss(p, y);
            delete p;

            b->set_val(i, b_val);

            float num_grad = (right_loss - left_loss) / (2.0f * ZERO_NN_EPSILON);
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

        if ((agg_grad_diff) / (agg_ana_grad + agg_num_grad) > ZERO_NN_EPSILON)
        {
            ZERO_CORE_THROW_ERROR("MODEL GRADIENTS VALIDATION FAILED");
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
    this->add_layer(new Linear(this->shared_params_, this->output_shape(), Shape(this->batch_size(), out_feature_cnt), activation));
}

void Model::linear(Shape y_shape, ActivationType activation)
{
    this->add_layer(new Linear(this->shared_params_, this->output_shape(), y_shape, activation));
}

void Model::linear(int batch_size, int in_feature_cnt, int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new Linear(this->shared_params_, Shape(batch_size, in_feature_cnt), Shape(batch_size, out_feature_cnt), activation));
}

void Model::linear(Shape in_shape, int out_feature_cnt, ActivationType activation)
{
    this->add_layer(new Linear(this->shared_params_, in_shape, Shape(in_shape[0], out_feature_cnt), activation));
}

void Model::conv2d(Shape filter_shape, ActivationType activation)
{
    this->add_layer(new Conv2d(this->shared_params_, this->output_shape(), filter_shape, Stride{1, 1}, activation));
}

void Model::conv2d(Shape filter_shape, Stride stride, ActivationType activation)
{
    this->add_layer(new Conv2d(this->shared_params_, this->output_shape(), filter_shape, stride, activation));
}

void Model::conv2d(Shape in_shape, Shape filter_shape, Stride stride, ActivationType activation)
{
    this->add_layer(new Conv2d(this->shared_params_, in_shape, filter_shape, stride, activation));
}

void Model::hadamard_product(int filter_cnt, ActivationType activation)
{
    this->add_layer(new HadamardProduct(this->shared_params_, this->output_shape(), filter_cnt, activation));
}

void Model::hadamard_product(Shape in_shape, int filter_cnt, ActivationType activation)
{
    this->add_layer(new HadamardProduct(this->shared_params_, in_shape, filter_cnt, activation));
}

void Model::matrix_product(int filter_cnt, ActivationType activation)
{
    this->add_layer(new MatrixProduct(this->shared_params_, this->output_shape(), filter_cnt, activation));
}

void Model::matrix_product(Shape in_shape, int filter_cnt, ActivationType activation)
{
    this->add_layer(new MatrixProduct(this->shared_params_, in_shape, filter_cnt, activation));
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
        }
    }

    return params;
}

void Model::share_parameters(std::vector<Parameters *> params)
{
    int param_idx = 0;
    for (Layer *lyr : this->lyrs_)
    {
        if (Learnable *lrn_lyr = dynamic_cast<Learnable *>(lyr))
        {
            lrn_lyr->share_parameters(params[param_idx]);
        }

        param_idx++;
    }
}

void Model::save_parameters(const char *file)
{
    auto params = this->parameters();

    for (auto p : params)
    {
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

void Model::change_batch_size(int batch_size)
{
    for (Layer *lyr : this->lyrs_)
    {
        lyr->change_batch_size(batch_size);
    }
}

Optimizer *Model::optimizer()
{
    return this->optim_;
}

void Model::performance_check(Tensor *x, Tensor *y, int epoch_cnt)
{
    CudaStopWatch *sw = new CudaStopWatch();

    sw->start();

    for (int i = 0; i < epoch_cnt; i++)
    {
        Tensor *p = this->forward(x);
        this->loss(p, y);
        this->backward(p, y);
        this->step();
        delete p;
    }

    sw->stop();

    sw->print_elapsed_seconds();

    delete sw;
}
