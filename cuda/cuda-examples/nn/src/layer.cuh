#include <ndarray.cuh>

#define THREADS_PER_BLOCK 32

class Layer
{
public:
    virtual void forward(ArrayNd *out) = 0;
    virtual ArrayNd *backward(ArrayNd *in) = 0;

    virtual ArrayNd *n() = 0;
};

class LinearLayer : public Layer
{
private:
    Array2d *n_;
    Array2d *w_;
    Array1d *b_;
    Array2d *dw_;
    Array1d *db_;

public:
    LinearLayer(int in_cnt, int out_cnt);
    ~LinearLayer();

    virtual void forward(Array2d *out);
    virtual Array2d *backward(Array2d *in);

    virtual ArrayNd *n();
};

class Activator
{
public:
    virtual void evaluate(Array2d *in, Array2d *out) = 0;
    virtual void derive(Array2d *in, Array2d *out) = 0;
};

class SigmoidActivator : public Activator
{
public:
    virtual void evaluate(Array2d *in, Array2d *out);
    virtual void derive(Array2d *in, Array2d *out);
};

class ActivationLayer : public Layer
{
private:
    Array2d *n_;
    Activator *a_;

public:
    ActivationLayer(Activator *a, int in_cnt);
    ~ActivationLayer();

    virtual void forward(Array2d *out);
    virtual Array2d *backward(Array2d *in);

    virtual ArrayNd *n();
};
