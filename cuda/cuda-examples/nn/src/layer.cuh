#include <tensor.cuh>

#define THREADS_PER_BLOCK 32

class Layer
{
public:
    virtual void forward(ArrayNd *out) = 0;
    virtual ArrayNd *backward(ArrayNd *d_l) = 0;
};

class LinearLayer : public Layer
{
private:
    Array2d *n_;
    Array2d *w_;
    Array1d *b_;

public:
    LinearLayer(int in_cnt, int out_cnt);
    ~LinearLayer();

    virtual void forward(Array2d *out);
    virtual Array2d *backward(Array2d *d_l);
};

class Activation
{
public:
    virtual void evaluate(Array2d *in, Array2d *out) = 0;
    virtual void derive(Array2d *in, Array2d *out) = 0;
};

class SigmoidActivation : public Activation
{
public:
    virtual void evaluate(Array2d *in, Array2d *out);
    virtual void derive(Array2d *in, Array2d *out);
};

class ActivationLayer : public Layer
{
private:
    Array2d *n_;
    Activation *a_;

public:
    ActivationLayer(int in_cnt);
    ~ActivationLayer();

    virtual void forward(Array2d *out);
    virtual Array2d *backward(Array2d *d_l);
};
