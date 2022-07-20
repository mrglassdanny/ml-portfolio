#include <tensor.cuh>

#define THREADS_PER_BLOCK 32

class Layer
{
protected:
    Tensor *n;

public:
    Layer();
    ~Layer();

    virtual void forward(Tensor *out) = 0;
    virtual Tensor *backward(Tensor *d_l) = 0;
};

class LinearLayer : public Layer
{
private:
    Tensor *w;
    Tensor *b;

public:
    LinearLayer(int in_cnt, int out_cnt);
    ~LinearLayer();

    void forward(Tensor *out);
    Tensor *backward(Tensor *d_l);
};
