#pragma once

#include "../core/mod.cuh"

#include "constants.cuh"

#include "layer/mod.cuh"
#include "loss/mod.cuh"
#include "optim/mod.cuh"

namespace nn
{
    using namespace nn::layer;
    using namespace nn::loss;
    using namespace nn::optim;

    class ERNN;

    class EnhancedResidual : public Linear
    {
    private:
        std::vector<Parameters *> residual_params_;

    public:
        EnhancedResidual(Shape in_shape, Shape out_shape, ActivationType activation);
        ~EnhancedResidual();

        void evaluate_residual(NdArray *out, int idx);

        void compile(ERNN *ernn, int my_idx);

        std::vector<Parameters *> residual_parameters();
    };

    class ERNN
    {
    private:
        std::vector<EnhancedResidual *> lyrs_;
        Loss *loss_;
        Optimizer *optim_;

    public:
        ERNN();
        ~ERNN();

        NdArray *forward(NdArray *x);
        float loss(NdArray *p, NdArray *y);
        float accuracy(NdArray *p, NdArray *y);
        void backward(NdArray *p, NdArray *y);
        void step();

        Shape input_shape();
        Shape output_shape();

        void add_layer(EnhancedResidual *lyr);
        void set_loss(Loss *loss);
        void set_optimizer(Optimizer *optim);

        std::vector<EnhancedResidual *> layers();
        EnhancedResidual *first_layer();
        EnhancedResidual *last_layer();
        std::vector<Parameters *> parameters();

        int batch_size();

        void compile();
    };
}