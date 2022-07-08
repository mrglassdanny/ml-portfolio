#include "tensor.cuh"

Shape::Shape()
{
}

Shape::Shape(int dim_1)
{
    this->dims.push_back(dim_1);
}

Shape::Shape(int dim_1, int dim_2)
{
    this->dims.push_back(dim_1);
    this->dims.push_back(dim_2);
}

Shape::Shape(int dim_1, int dim_2, int dim_3)
{
    this->dims.push_back(dim_1);
    this->dims.push_back(dim_2);
    this->dims.push_back(dim_3);
}

Shape::Shape(int dim_1, int dim_2, int dim_3, int dim_4)
{
    this->dims.push_back(dim_1);
    this->dims.push_back(dim_2);
    this->dims.push_back(dim_3);
    this->dims.push_back(dim_4);
}

Shape::Shape(std::vector<int> dims)
{
    this->dims = dims;
}

Shape::~Shape()
{
}

void Shape::print()
{
    int dims_cnt = this->get_dims_cnt();

    for (int i = 0; i < dims_cnt; i++)
    {
        printf("%d", this->dims[i]);

        if (i < dims_cnt - 1)
        {
            printf("x");
        }
    }

    printf("\n");
}

int Shape::get_dim(int dim_idx)
{
    return this->dims[dim_idx];
}

int Shape::get_dims_cnt()
{
    return this->dims.size();
}

int Shape::get_dims_size()
{
    int dims_size = 1;

    for (int i = 0; i < this->get_dims_cnt(); i++)
    {
        dims_size *= this->dims[i];
    }

    return dims_size;
}

Tensor::Tensor(Tensor &src)
{
    this->cuda_flg = src.cuda_flg;
    this->shape = src.shape;

    size_t size = sizeof(float) * this->shape.get_dims_size();

    if (src.cuda_flg)
    {
        cudaMalloc(&this->data, size);
        cudaMemcpy(this->data, src.data, size, cudaMemcpyDeviceToDevice);
    }
    else
    {
        this->data = (float *)malloc(size);
        memcpy(this->data, src.data, size);
    }
}

Tensor::Tensor(bool cuda_flg, Shape shape)
{
    this->cuda_flg = cuda_flg;
    this->shape = shape;

    size_t size = sizeof(float) * this->shape.get_dims_size();

    if (cuda_flg)
    {
        cudaMalloc(&this->data, size);
    }
    else
    {
        this->data = (float *)malloc(size);
    }
}

Tensor::~Tensor()
{
    if (this->cuda_flg)
    {
        cudaFree(this->data);
    }
    else
    {
        free(this->data);
    }
}

void Tensor::print()
{
    bool orig_cuda_flg = this->cuda_flg;

    this->to_cpu();

    switch (this->shape.get_dims_cnt())
    {
    case 1:
    {
        int dim_1 = this->shape.get_dim(0);
        printf("[ ");
        for (int i = 0; i < dim_1; i++)
        {
            if (i == dim_1 - 1)
            {
                printf("%f", this->data[i]);
            }
            else
            {
                printf("%f, ", this->data[i]);
            }
        }
        printf(" ]");
    }

    break;
    case 2:
    {
        int dim_1 = this->shape.get_dim(0);
        int dim_2 = this->shape.get_dim(1);

        printf("[");
        for (int i = 0; i < dim_1; i++)
        {

            if (i == 0)
            {
                printf(" [ ");
            }
            else
            {
                printf("  [ ");
            }

            for (int j = 0; j < dim_2; j++)
            {
                if (j == dim_2 - 1)
                {
                    printf("%f", this->data[i * dim_2 + j]);
                }
                else
                {
                    printf("%f, ", this->data[i * dim_2 + j]);
                }
            }

            if (i == dim_1 - 1)
            {
                printf(" ] ");
            }
            else
            {
                printf(" ],\n");
            }
        }
        printf("]\n");
    }
    break;
    case 3:
    {
        int dim_1 = this->shape.get_dim(0);
        int dim_2 = this->shape.get_dim(1);
        int dim_3 = this->shape.get_dim(2);

        printf("[");
        for (int i = 0; i < dim_1; i++)
        {

            if (i == 0)
            {
                printf(" [ ");
            }
            else
            {
                printf("  [ ");
            }

            for (int j = 0; j < dim_2; j++)
            {

                if (j == 0)
                {
                    printf(" [ ");
                }
                else
                {
                    printf("  [ ");
                }

                for (int k = 0; k < dim_3; k++)
                {
                    if (k == dim_3 - 1)
                    {
                        printf("%f", this->data[(i * dim_2 * dim_3) + (j * dim_3) + k]);
                    }
                    else
                    {
                        printf("%f, ", this->data[(i * dim_2 * dim_3) + (j * dim_3) + k]);
                    }
                }

                if (j == dim_2 - 1)
                {
                    printf(" ] ");
                }
                else
                {
                    printf(" ],\n");
                }
            }

            if (i == dim_1 - 1)
            {
                printf(" ] ");
            }
            else
            {
                printf(" ],\n");
            }
        }
        printf("]\n");
    }
    break;
    default:
        for (int i = 0; i < this->shape.get_dims_size(); i++)
        {
            printf("%d: %f\n", i, this->data[i]);
        }
        break;
    }

    printf("\n");

    if (orig_cuda_flg)
    {
        this->to_cuda();
    }
}

Shape Tensor::get_shape()
{
    return this->shape;
}

void Tensor::to_cpu()
{
    if (this->cuda_flg)
    {
        size_t size = sizeof(float) * this->shape.get_dims_size();
        float *dst = (float *)malloc(size);
        cudaMemcpy(dst, this->data, size, cudaMemcpyDeviceToHost);
        cudaFree(this->data);
        this->data = dst;
        this->cuda_flg = false;
    }
}

void Tensor::to_cuda()
{
    if (!this->cuda_flg)
    {
        size_t size = sizeof(float) * this->shape.get_dims_size();
        float *dst;
        cudaMalloc(&dst, size);
        cudaMemcpy(dst, this->data, size, cudaMemcpyHostToDevice);
        free(this->data);
        this->data = dst;
        this->cuda_flg = true;
    }
}

void Tensor::zeros()
{
    size_t size = sizeof(float) * this->shape.get_dims_size();

    if (this->cuda_flg)
    {
        cudaMemset(this->data, 0, size);
    }
    else
    {
        memset(this->data, 0, size);
    }
}

void Tensor::rands(float mean, float stddev)
{
    bool orig_cuda_flg = this->cuda_flg;

    this->to_cpu();

    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < this->shape.get_dims_size(); i++)
        {
            std::normal_distribution<float> d(mean, stddev);
            this->data[i] = d(gen);
        }
    }

    if (orig_cuda_flg)
    {
        this->to_cuda();
    }
}
