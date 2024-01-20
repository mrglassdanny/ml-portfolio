
#include "flatten.h"

namespace tallgeese
{
    namespace nn
    {
        namespace layer
        {
            Flatten::Flatten(ADContext *ctx, Shape input_shape)
                : Layer(ctx)
            {
                int inputs = 1;
                for (int i = 1; i < input_shape.size(); i++)
                {
                    inputs *= input_shape[i];
                }

                this->y = Tensor::zeros({input_shape[0], inputs});
            }

            Flatten::~Flatten()
            {
                delete this->y;
            }

            void Flatten::reset()
            {
                Layer::reset();
            }

            Tensor *Flatten::forward(Tensor *x)
            {
                this->reset();

                this->y->copy_data(x);

                return this->y;
            }
        }
    }
}