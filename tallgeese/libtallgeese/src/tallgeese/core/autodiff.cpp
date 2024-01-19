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

        IntVar::IntVar(Operation op)
        {
            this->op = op;
        }

        IntVar::IntVar(Operation op, float pd1, float pd2)
        {
            this->op = op;
            this->pds[0] = pd1;
            this->pds[1] = pd2;
        }

        IntVar::IntVar(Operation op, float pd1, float pd2, int i1, int i2)
        {
            this->op = op;
            this->pds[0] = pd1;
            this->pds[1] = pd2;
            this->is[0] = i1;
            this->is[1] = i2;
        }

        void IntVar::print()
        {
            switch (this->op)
            {
            case Add:
                printf("Add\t");
                break;
            case Multiply:
                printf("Multiply\t");
                break;
            case Power:
                printf("Power\t");
                break;
            case Sigmoid:
                printf("Sigmoid\t");
                break;
            default:
                break;
            }

            printf("dy/dx: %f\n", this->d);
        }

        Tensor::Tensor(std::vector<int> shape)
        {
            this->shape = shape;
            this->data = new Var[this->count()];
            this->zeros();
        }

        Tensor::Tensor(std::vector<int> shape, float val)
        {
            this->shape = shape;
            this->data = new Var[this->count()];
            this->fill(val);
        }

        Tensor::Tensor(std::vector<int> shape, float mean, float stddev)
        {
            this->shape = shape;
            this->data = new Var[this->count()];
            this->random(mean, stddev);
        }

        Tensor::~Tensor()
        {
            delete this->data;
        }

        Tensor *Tensor::zeros(std::vector<int> shape)
        {
            auto t = new Tensor(shape);
            t->zeros();
            return t;
        }

        Tensor *Tensor::fill(std::vector<int> shape, float val)
        {
            auto t = new Tensor(shape);
            t->fill(val);
            return t;
        }

        Tensor *Tensor::random(std::vector<int> shape)
        {
            auto t = new Tensor(shape);
            t->random();
            return t;
        }

        Tensor *Tensor::random(std::vector<int> shape, float mean, float stddev)
        {
            auto t = new Tensor(shape);
            t->random(mean, stddev);
            return t;
        }

        void Tensor::print()
        {
            printf("SHAPE: ");
            for (int i = 0; i < this->dims(); i++)
            {
                if (i == this->dims() - 1)
                    printf("%d\n", this->shape[i]);
                else
                    printf("%dx", this->shape[i]);
            }

            printf("DATA:\n");
            switch (this->dims())
            {
            case 1:
                for (int i = 0; i < this->count(); i++)
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
                        this->data[i * this->shape[1] + j].print();
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

        bool Tensor::has_same_shape(Tensor *other)
        {
            return this->shape == other->shape;
        }

        int Tensor::dims()
        {
            return this->shape.size();
        }

        int Tensor::count()
        {
            int num = 1;
            for (auto dim : this->shape)
            {
                num *= dim;
            }
            return num;
        }

        size_t Tensor::size()
        {
            return this->count() * sizeof(Var);
        }

        void Tensor::zeros()
        {
            this->fill(0.0f);
        }

        void Tensor::fill(float val)
        {
            for (int i = 0; i < this->count(); i++)
            {
                this->data[i] = Var(val);
            }
        }

        void Tensor::random()
        {
            this->random(0.0f, 1.0f);
        }

        void Tensor::random(float mean, float stddev)
        {
            std::random_device rd;
            std::mt19937 gen(rd());

            for (int i = 0; i < this->count(); i++)
            {
                std::normal_distribution<float> d(mean, stddev);
                this->data[i].v = d(gen);
            }
        }

        ADContext::ADContext()
        {
        }

        ADContext::ADContext(bool trace)
        {
            this->trace = trace;
        }

        void ADContext::validate_shapes_are_same(Tensor *a, Tensor *b)
        {
            if (!a->has_same_shape(b))
            {
                TALLGEESE_CORE_THROW_ERROR("AUTODIFF: Shapes do not match");
            }
        }

        Var ADContext::op(Operation op, float v, float pd1, float pd2, int i1, int i2)
        {
            if (!this->replaying)
            {
                Var var(v, this->tape.size());
                this->tape.push_back(IntVar(op, pd1, pd2, i1, i2));

                if (this->trace)
                    this->vars.push_back(var);

                return var;
            }
            else
            {
                Var var(v);
                return var;
            }
        }

        Var ADContext::var(float v)
        {
            Var var(v, this->tape.size());
            this->tape.push_back(IntVar(None));

            if (this->trace)
                this->vars.push_back(var);

            return var;
        }

        Tensor *ADContext::var(Tensor *tensor)
        {
            for (int i = 0; i < tensor->count(); i++)
            {
                tensor->data[i] = this->var(tensor->data[i].v);
            }
            return tensor;
        }

        Var ADContext::parm(float v)
        {
            Var var(v, this->tape.size());
            this->tape.push_back(IntVar(Parameter));

            if (this->trace)
                this->vars.push_back(var);

            return var;
        }

        Tensor *ADContext::parm(Tensor *tensor)
        {
            for (int i = 0; i < tensor->count(); i++)
            {
                tensor->data[i] = this->parm(tensor->data[i].v);
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

        Var ADContext::evaluate()
        {
            if (!this->trace)
            {
                printf("AUTODIFF EVALUATION: NOT TRACING\n");
                return Var(0.0f);
            }

            this->replaying = true;

            for (int i = 0; i < this->tape.size(); i++)
            {
                auto iv = this->tape[i];
                switch (iv.op)
                {
                case Add:
                    this->vars[i].v = this->add(this->vars[iv.is[0]], this->vars[iv.is[1]]).v;
                    break;
                case Multiply:
                    this->vars[i].v = this->multiply(this->vars[iv.is[0]], this->vars[iv.is[1]]).v;
                    break;
                case Power:
                    this->vars[i].v = this->power(this->vars[iv.is[0]], this->vars[iv.is[1]]).v;
                    break;
                case Sigmoid:
                    this->vars[i].v = this->sigmoid(this->vars[iv.is[0]]).v;
                    break;
                default:
                    break;
                }
            }

            this->replaying = false;

            return this->vars[this->vars.size() - 1];
        }

        void ADContext::reset_gradients()
        {
            for (auto t : this->tape)
            {
                t.d = 0.0f;
            }
        }

        void ADContext::check_gradients()
        {
            if (!this->trace)
            {
                printf("AUTODIFF GRADIENT CHECK: NOT TRACING\n");
                return;
            }

            float agg_ana_grad = 0.0f;
            float agg_num_grad = 0.0f;
            float agg_grad_diff = 0.0f;

            std::vector<Var> o_vars = this->vars;

            for (int i = 0; i < this->tape.size(); i++)
            {
                // Only want to look at parameters.
                if (this->tape[i].op == Parameter)
                {
                    Var o_var = this->vars[i];

                    this->vars[i].v = o_var.v - TALLGEESE_CORE_EPSILON;
                    Var l_var = this->evaluate();

                    this->vars[i].v = o_var.v + TALLGEESE_CORE_EPSILON;
                    Var r_var = this->evaluate();

                    float num_grad = (r_var.v - l_var.v) / (2.0f * TALLGEESE_CORE_EPSILON);
                    float ana_grad = this->tape[i].d;

                    printf("NUM: %f\n", num_grad);
                    printf("ANA: %f\n\n", ana_grad);

                    agg_ana_grad += (ana_grad * ana_grad);
                    agg_num_grad += (num_grad * num_grad);
                    agg_grad_diff += ((ana_grad - num_grad) * (ana_grad - num_grad));

                    this->vars = o_vars;
                }
            }

            if ((agg_grad_diff) == 0.0f && (agg_ana_grad + agg_num_grad) == 0.0f)
            {
                printf("AUTODIFF GRADIENT CHECK RESULT: %f\n", 0.0f);
            }
            else
            {
                printf("AUTODIFF GRADIENT CHECK RESULT: %f\n", (agg_grad_diff) / (agg_ana_grad + agg_num_grad));

                if ((agg_grad_diff) / (agg_ana_grad + agg_num_grad) > TALLGEESE_CORE_EPSILON)
                {
                    printf("AUTODIFF GRADIENT CHECK FAILED");
                }
            }
        }

        Var ADContext::add(Var a, Var b)
        {
            return this->op(Add, a.v + b.v, 1.0f, 1.0f, a.i, b.i);
        }

        Var ADContext::multiply(Var a, Var b)
        {
            return this->op(Multiply, a.v * b.v, b.v, a.v, a.i, b.i);
        }

        Var ADContext::power(Var a, Var b)
        {
            return this->op(Power, pow(a.v, b.v), b.v * pow(a.v, b.v - 1), pow(a.v, b.v) * log(a.v), a.i, b.i);
        }

        Var ADContext::sigmoid(Var a)
        {
            auto v = (1.0f / (1.0f + exp(-a.v)));
            return this->op(Sigmoid, v, (v) * (1.0f - v), 0.0f, a.i, TALLGEESE_CORE_INVALID_INTVAR_INDEX);
        }

        Var ADContext::dot(Tensor *a, Tensor *b)
        {
            this->validate_shapes_are_same(a, b);

            Var d = this->var(0.0f);
            for (int i = 0; i < a->count(); i++)
            {
                auto c = this->multiply(a->data[i], b->data[i]);
                d = this->add(c, d);
            }
            return d;
        }

        Tensor *ADContext::dot(Tensor *a, Tensor *b, Tensor *c)
        {
            this->validate_shapes_are_same(a, b);

            switch (a->dims())
            {
            case 1:
                c->data[0] = this->dot(a, b);
                break;
            case 2:
                for (int i = 0; i < a->shape[0]; i++)
                {
                    Var vd = this->var(0.0f);
                    for (int j = 0; j < a->shape[1]; j++)
                    {
                        auto vc = this->multiply(a->data[i * a->shape[1] + j], b->data[i * a->shape[1] + j]);
                        vd = this->add(vc, vd);
                    }
                    c->data[i] = vd;
                }
                break;
            }

            return c;
        }

        Tensor *ADContext::matrix_multiply(Tensor *x, Tensor *w, Tensor *y)
        {
            int x_rows = x->shape[0];
            int x_cols = x->shape[1];

            int w_rows = w->shape[0];
            int w_cols = w->shape[1];

            /*
                Matrix multiplication is only valid if the number of columns of the first matrix
                are equal to the number of rows of the second matrix; further, the resulting
                matrix will have the number of rows of the first matrix and the number of columns of
                the second matrix.
            */

            int y_rows = x_rows;
            int y_cols = w_cols;

            if (x_cols != w_rows)
            {
                TALLGEESE_CORE_THROW_ERROR("AUTODIFF: Incompatible matrix shapes for matrix multiply");
            }

            if (y->shape[0] != y_rows || y->shape[1] != y_cols)
            {
                TALLGEESE_CORE_THROW_ERROR("AUTODIFF: Incompatible output matrix shape for matrix multiply");
            }

            for (int y_row = 0; y_row < y_rows; y_row++)
            {
                int x_row = y_row;
                for (int y_col = 0; y_col < y_cols; y_col++)
                {
                    int w_col = y_col;
                    for (int x_col = 0; x_col < x_cols; x_col++)
                    {
                        int w_row = x_col;
                        y->data[y_row * y_cols + y_col] = this->add(
                            this->multiply(x->data[x_row * x_cols + x_col], w->data[w_row * w_cols + w_col]),
                            y->data[y_row * y_cols + y_col]);
                    }
                }
            }

            return y;
        }

        Tensor *ADContext::sigmoid(Tensor *a, Tensor *b)
        {
            this->validate_shapes_are_same(a, b);

            for (int i = 0; i < a->count(); i++)
            {
                b->data[i] = this->sigmoid(a->data[i]);
            }
            return b;
        }
    }
}