#include <tensor.cuh>

#define THREADS_PER_BLOCK 32

class Layer
{
protected:
    Tensor *n;

public:
    Layer();
    ~Layer();

    virtual void evaluate(Tensor *out) = 0;
    virtual Tensor *derive(Tensor *d_l) = 0;
};

class LinearLayer : public Layer
{
private:
    Tensor *w;
    Tensor *b;

public:
    LinearLayer();
    ~LinearLayer();

    void evaluate(Tensor *out);
    Tensor *derive(Tensor *d_l);
};
