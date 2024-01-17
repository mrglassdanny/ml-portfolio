#include "autodiff.h"

namespace tallgeese
{
    namespace core
    {
        Var::Var() {}

        Var::Var(float v)
        {
            this->v = v;
        }

        Var::Var(float v, int i)
        {
            this->v = v;
            this->i = i;
        }

        void Var::print()
        {
            printf("%f", this->v);
        }

        IntVar::IntVar() {}

        IntVar::IntVar(float pd1, float pd2)
        {
            this->pds[0] = pd1;
            this->pds[1] = pd2;
        }

        IntVar::IntVar(float pd1, float pd2, int i1, int i2)
        {
            this->pds[0] = pd1;
            this->pds[1] = pd2;
            this->is[0] = i1;
            this->is[1] = i2;
        }

        Tensor::Tensor(std::vector<int> shape)
        {
            this->shape = shape;
            this->data = new Var[this->size()];
            this->zeros();
        }

        Tensor::Tensor(std::vector<int> shape, float val)
        {
            this->shape = shape;
            this->data = new Var[this->size()];
            this->fill(val);
        }

        Tensor::Tensor(std::vector<int> shape, float mean, float stddev)
        {
            this->shape = shape;
            this->data = new Var[this->size()];
            this->random(mean, stddev);
        }

        Tensor::~Tensor()
        {
            delete this->data;
        }

        void Tensor::print()
        {
            printf("SHAPE: ");
            for (int i = 0; i < this->count(); i++)
            {
                if (i == this->shape.size() - 1)
                    printf("%d\n", this->shape[i]);
                else
                    printf("%dx", this->shape[i]);
            }

            printf("DATA: ");
            switch (this->shape.size())
            {
            case 1:
                for (int i = 0; i < this->size(); i++)
                {
                    this->data[i].print();
                    printf("\t");
                }
                break;
            case 2:
                for (int i = 0; i < this->shape[0]; i++)
                {
                    for (int j = 0; j < this->shape[1]; j++)
                    {
                        this->data[i].print();
                        printf("\t");
                    }
                    printf("\n");
                }
                break;
            default:
                for (int i = 0; i < this->count(); i++)
                {
                    this->data[i].print();
                    printf("\t");
                }
                break;
            }

            printf("\n\n");
        }

        Var *Tensor::get_data()
        {
            return this->data;
        }

        int Tensor::count()
        {
            return this->shape.size();
        }

        int Tensor::size()
        {
            int num = 1;
            for (auto dim : this->shape)
            {
                num *= dim;
            }
            return num;
        }

        void Tensor::zeros()
        {
            this->fill(0.0f);
        }

        void Tensor::fill(float val)
        {
            for (int i = 0; i < this->size(); i++)
            {
                this->data[i] = Var(val);
            }
        }

        void Tensor::random(float mean, float stddev)
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            for (int i = 0; i < this->size(); i++)
            {
                std::normal_distribution<float> d(mean, stddev);
                auto n = Var(d(gen));
                this->data[i] = n;
            }
        }

        ADContext::ADContext()
        {
        }

        Var ADContext::op(float v, float pd1, float pd2, int i1, int i2)
        {
            Var var(v, this->tape.size());
            this->tape.push_back(IntVar(pd1, pd2, i1, i2));
            return var;
        }

        Var ADContext::var(float v)
        {
            Var var(v, this->tape.size());
            this->tape.push_back(IntVar());
            return var;
        }

        Tensor *ADContext::tensor(Tensor *tensor)
        {
            for (int i = 0; i < tensor->size(); i++)
            {
                tensor->get_data()[i] = this->var(tensor->get_data()[i].v);
            }
            return tensor;
        }

        void ADContext::derive()
        {
            this->tape[this->tape.size() - 1].d = 1.0f;
            for (int i = this->tape.size() - 1; i >= 0; i--)
            {
                if (this->tape[i].is[0] != TALLGEESE_CORE_INVALID_INTVAR_INDEX)
                    this->tape[this->tape[i].is[0]].d += this->tape[i].d * this->tape[i].pds[0];
                if (this->tape[i].is[1] != TALLGEESE_CORE_INVALID_INTVAR_INDEX)
                    this->tape[this->tape[i].is[1]].d += this->tape[i].d * this->tape[i].pds[1];
            }
        }

        float ADContext::get_derivative(Var var)
        {
            return this->tape[var.i].d;
        }

        Var ADContext::add(Var a, Var b)
        {
            return this->op(a.v + b.v, 1.0f, 1.0f, a.i, b.i);
        }

        Var ADContext::mul(Var a, Var b)
        {
            return this->op(a.v * b.v, b.v, a.v, a.i, b.i);
        }

        Var ADContext::pwr(Var a, float b)
        {
            return this->op(pow(a.v, b), b * pow(a.v, b - 1), 0.0f, a.i, TALLGEESE_CORE_INVALID_INTVAR_INDEX);
        }

        void ADContext::sum(Tensor *a, Tensor *b)
        {
            switch (b->count())
            {
            }
        }
    }
}