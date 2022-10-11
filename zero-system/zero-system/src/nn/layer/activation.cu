#include "activation.cuh"

using namespace nn::layer;

void summarize_activation(Activation activation)
{
    printf("Activation: ");
    
    switch (activation)
    {
    case Activation::None:
        printf("None");
        break;
    case Activation::Sigmoid:
        printf("Sigmoid");
        break;
    case Activation::Tanh:
        printf("Tanh");
        break;
    case Activation::ReLU:
        printf("ReLU");
        break;
    default: // None
        break;
    }
}