#include <tensor.cuh>

#define THREADS_PER_BLOCK 32

class Layer
{
protected:
    ArrayNd *n;

public:
    Layer();
    ~Layer();

    virtual void forward(ArrayNd *out) = 0;
    virtual ArrayNd *backward(ArrayNd *d_l) = 0;
};

class LinearLayer : public Layer
{
private:
    Array2d *w;
    Array1d *b;

public:
    LinearLayer(int in_cnt, int out_cnt);
    ~LinearLayer();

    void forward(Array2d *out);
    Array2d *backward(Array2d *d_l);
};
