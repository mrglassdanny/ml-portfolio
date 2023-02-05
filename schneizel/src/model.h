#pragma once

#include <stdlib.h>
#include <string.h>

#include <vector>
#include <random>

namespace schneizel
{
    namespace model
    {
        class Layer
        {
        private:
            int fan_in;
            int fan_out;
            bool activation;
            float *n;
            float *dn;
            float *w;
            float *dw;
            float *b;
            float *db;

        public:
            Layer(int fan_in, int fan_out, bool activation);
            ~Layer();

            Layer *copy();

            int inputs();
            int outputs();

            float *neurons();
            float *neuron_grads();
            void copy_neurons(float *x);

            float *weights();
            float *biases();

            float *weight_grads();
            float *bias_grads();

            void forward(float *out);
            void backward(float *in, float *in_n);

            void zero_grad();
        };

        class Model
        {
        private:
            std::vector<Layer *> layers;
            float learning_rate;

        public:
            Model(float learning_rate);
            ~Model();

            Model *copy();

            void add_layer(Layer *layer);

            float forward(float *x);
            void backward(float p, float y);
            void step();
            float loss(float p, float y);

            void grad_check();
        };

        void init(const char* params_path, int thread_cnt);
        Model *get_model(int thread_id);
    }

}