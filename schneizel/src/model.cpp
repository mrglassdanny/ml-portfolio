#include "model.h"

namespace schneizel
{
    namespace model
    {
        Layer::Layer(int fan_in, int fan_out, bool activation)
            : fan_in(fan_in), fan_out(fan_out), activation(activation)
        {
            this->n = (float *)malloc(sizeof(float) * fan_in);
            this->dn = (float *)malloc(sizeof(float) * fan_in);
            this->w = (float *)malloc(sizeof(float) * (fan_in * fan_out));
            this->dw = (float *)malloc(sizeof(float) * (fan_in * fan_out));
            this->b = (float *)malloc(sizeof(float) * fan_out);
            this->db = (float *)malloc(sizeof(float) * fan_out);

            // Weight initialization:
            {
                std::random_device rd;
                std::mt19937 gen(rd());

                for (int i = 0; i < fan_in * fan_out; i++)
                {
                    std::normal_distribution<float> d(0.0f, sqrt(2.0f / fan_in));
                    this->w[i] = d(gen);
                }
            }
            memset(this->b, 0, sizeof(float) * fan_out);

            this->zero_grad();
        }

        Layer::~Layer()
        {
            free(this->n);
            free(this->w);
            free(this->b);
            free(this->dn);
            free(this->dw);
            free(this->db);
        }

        int Layer::inputs()
        {
            return this->fan_in;
        }

        int Layer::outputs()
        {
            return this->fan_out;
        }

        float *Layer::neurons()
        {
            return this->n;
        }

        float *Layer::neuron_grads()
        {
            return this->dn;
        }

        float *Layer::weights()
        {
            return this->w;
        }

        float *Layer::weight_grads()
        {
            return this->dw;
        }

        float *Layer::biases()
        {
            return this->b;
        }

        float *Layer::bias_grads()
        {
            return this->db;
        }

        void Layer::copy_neurons(float *x)
        {
            memcpy(this->n, x, sizeof(float) * this->fan_in);
        }

        void Layer::forward(float *out)
        {
            memset(out, 0, sizeof(float) * this->fan_out);

            for (int i = 0; i < this->fan_out; i++)
            {
                for (int j = 0; j < this->fan_in; j++)
                {
                    out[i] += (this->n[j] * this->w[i * this->fan_in + j]);
                }
                out[i] += this->b[i];

                if (this->activation)
                {
                    // Tanh
                    out[i] = ((exp(out[i]) - exp(-out[i])) / (exp(out[i]) + exp(-out[i])));
                }
            }
        }

        void Layer::backward(float *in, float *in_n)
        {
            if (this->activation)
            {
                for (int i = 0; i < this->fan_out; i++)
                {
                    // Tanh
                    in[i] *= (1.0f - (in_n[i] * in_n[i]));
                }
            }

            for (int i = 0; i < this->fan_out; i++)
            {
                for (int j = 0; j < this->fan_in; j++)
                {
                    this->dw[i * this->fan_in + j] += (in[i] * this->n[j]);
                }
                this->db[i] += in[i];
            }

            for (int i = 0; i < this->fan_in; i++)
            {
                for (int j = 0; j < this->fan_out; j++)
                {
                    this->dn[i] += (in[j] * this->w[j * this->fan_in + i]);
                }
            }
        }

        void Layer::zero_grad()
        {
            memset(this->dn, 0, sizeof(float) * this->fan_in);
            memset(this->dw, 0, sizeof(float) * (this->fan_in * this->fan_out));
            memset(this->db, 0, sizeof(float) * this->fan_out);
        }

        Model::Model(float learning_rate)
            : learning_rate(learning_rate)
        {
        }

        Model::~Model()
        {
            for (auto lyr : this->layers)
            {
                delete lyr;
            }
        }

        void Model::add_layer(Layer *lyr)
        {
            this->layers.push_back(lyr);
        }

        float Model::forward(float *x)
        {
            this->layers[0]->copy_neurons(x);
            for (int i = 0; i < this->layers.size() - 1; i++)
            {
                this->layers[i]->forward(this->layers[i + 1]->neurons());
            }

            float p = 0.0f;
            this->layers[this->layers.size() - 1]->forward(&p);

            return p;
        }

        void Model::backward(float p, float y)
        {
            float dl = 2.0f * (p - y);
            float *dl_ptr = &dl;

            float *p_ptr = &p;

            for (int i = this->layers.size() - 1; i >= 0; i--)
            {
                this->layers[i]->backward(dl_ptr, p_ptr);
                dl_ptr = this->layers[i]->neuron_grads();
                p_ptr = this->layers[i]->neurons();
            }
        }

        void Model::step()
        {
            for (auto lyr : this->layers)
            {
                for (int i = 0; i < lyr->outputs(); i++)
                {
                    for (int j = 0; j < lyr->inputs(); j++)
                    {
                        lyr->weights()[i * lyr->inputs() + j] -= (this->learning_rate * lyr->weight_grads()[i * lyr->inputs() + j]);
                    }
                    lyr->biases()[i] -= (this->learning_rate * lyr->bias_grads()[i]);
                }

                lyr->zero_grad();
            }
        }

        float Model::loss(float p, float y)
        {
            float diff = p - y;
            return diff * diff;
        }

        void Model::grad_check()
        {
            float *x = (float *)malloc(sizeof(float) * this->layers[0]->inputs());
            {
                std::random_device rd;
                std::mt19937 gen(rd());

                for (int i = 0; i < this->layers[0]->inputs(); i++)
                {
                    std::normal_distribution<float> d(0.0f, 1.0f);
                    x[i] = d(gen);
                }
            }
            float y = 1.0f;

            for (auto lyr : this->layers)
            {
                lyr->zero_grad();
            }

            float agg_ana_grad = 0.0f;
            float agg_num_grad = 0.0f;
            float agg_grad_diff = 0.0f;

            float p = this->forward(x);
            this->backward(p, y);

            int lyr_idx = 0;
            for (auto lyr : this->layers)
            {
                float *w = lyr->weights();
                float *dw = lyr->weight_grads();
                float *b = lyr->biases();
                float *db = lyr->bias_grads();

                for (int i = 0; i < lyr->inputs() * lyr->outputs(); i++)
                {
                    float w_val = w[i];

                    w[i] = w_val - 0.001f;
                    p = this->forward(x);
                    float left_loss = this->loss(p, y);

                    w[i] = w_val + 0.001f;
                    p = this->forward(x);
                    float right_loss = this->loss(p, y);

                    w[i] = w_val;

                    float num_grad = (right_loss - left_loss) / (2.0f * 0.001f);
                    float ana_grad = dw[i];

                    printf("W: %d  %d\t|%f - %f| = %f\n", lyr_idx, i, ana_grad, num_grad, fabs(ana_grad - num_grad));

                    agg_ana_grad += (ana_grad * ana_grad);
                    agg_num_grad += (num_grad * num_grad);
                    agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));
                }

                for (int i = 0; i < lyr->outputs(); i++)
                {
                    float b_val = b[i];

                    b[i] = b_val - 0.001f;
                    p = this->forward(x);
                    float left_loss = this->loss(p, y);

                    b[i] = b_val + 0.001f;
                    p = this->forward(x);
                    float right_loss = this->loss(p, y);

                    b[i] = b_val;

                    float num_grad = (right_loss - left_loss) / (2.0f * 0.001f);
                    float ana_grad = db[i];

                    printf("B: %d  %d\t|%f - %f| = %f\n", lyr_idx, i, ana_grad, num_grad, fabs(ana_grad - num_grad));

                    agg_ana_grad += (ana_grad * ana_grad);
                    agg_num_grad += (num_grad * num_grad);
                    agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));
                }

                lyr_idx++;
            }

            if ((agg_grad_diff) == 0.0f && (agg_ana_grad + agg_num_grad) == 0.0f)
            {
                printf("GRADIENT CHECK RESULT: %f\n", 0.0f);
            }
            else
            {
                printf("GRADIENT CHECK RESULT: %f\n", (agg_grad_diff) / (agg_ana_grad + agg_num_grad));

                if ((agg_grad_diff) / (agg_ana_grad + agg_num_grad) > 0.001f)
                {
                    printf("MODEL GRADIENTS VALIDATION FAILED");
                }
            }

            free(x);
        }

        std::vector<Model *> models;

        void init(const char *params_path, int thread_cnt)
        {
            auto model = new Model(0.001f);
            model->add_layer(new Layer(64, 128, true));
            model->add_layer(new Layer(128, 128, true));
            model->add_layer(new Layer(128, 16, true));
            model->add_layer(new Layer(16, 1, false));

            // TODO
            if (params_path != nullptr)
            {
            }

            // TODO
            for (int i = 0; i < thread_cnt; i++)
            {
                models.push_back(model);
            }
        }

        Model *get_model(int thread_id)
        {
            return models[thread_id];
        }

    }
}