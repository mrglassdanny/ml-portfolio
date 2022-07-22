#include "tensor.cuh"

Dimensions::Dimensions()
{
}

Dimensions::Dimensions(int dim_1)
{
    this->dims_vec_.push_back(dim_1);
}

Dimensions::Dimensions(int dim_1, int dim_2)
{
    this->dims_vec_.push_back(dim_1);
    this->dims_vec_.push_back(dim_2);
}

Dimensions::Dimensions(int dim_1, int dim_2, int dim_3)
{
    this->dims_vec_.push_back(dim_1);
    this->dims_vec_.push_back(dim_2);
    this->dims_vec_.push_back(dim_3);
}

Dimensions::Dimensions(int dim_1, int dim_2, int dim_3, int dim_4)
{
    this->dims_vec_.push_back(dim_1);
    this->dims_vec_.push_back(dim_2);
    this->dims_vec_.push_back(dim_3);
    this->dims_vec_.push_back(dim_4);
}

Dimensions::Dimensions(std::vector<int> dims_vec)
{
    this->dims_vec_ = dims_vec;
}

Dimensions::~Dimensions()
{
}

void Dimensions::print()
{
    int cnt = this->count();

    for (int i = 0; i < cnt; i++)
    {
        printf("%d", this->dims_vec_[i]);

        if (i < cnt - 1)
        {
            printf("x");
        }
    }

    printf("\n");
}

int Dimensions::dim(int dim_idx)
{
    return this->dims_vec_[dim_idx];
}

int Dimensions::count()
{
    return this->dims_vec_.size();
}

int Dimensions::size()
{
    int size = 1;

    for (int i = 0; i < this->count(); i++)
    {
        size *= this->dims_vec_[i];
    }

    return size;
}

ArrayNd::ArrayNd(ArrayNd &src)
{
    this->cuda_ = src.cuda_;
    this->dims_ = src.dims_;

    size_t size = this->size();

    if (src.cuda_)
    {
        cudaMalloc(&this->data_, size);
        cudaMemcpy(this->data_, src.data_, size, cudaMemcpyDeviceToDevice);
    }
    else
    {
        this->data_ = (float *)malloc(size);
        memcpy(this->data_, src.data_, size);
    }
}

ArrayNd::ArrayNd(bool cuda, Dimensions dims)
{
    this->cuda_ = cuda;
    this->dims_ = dims;

    size_t size = this->size();

    if (cuda)
    {
        cudaMalloc(&this->data_, size);
    }
    else
    {
        this->data_ = (float *)malloc(size);
    }
}

ArrayNd::~ArrayNd()
{
    if (this->cuda_)
    {
        cudaFree(this->data_);
    }
    else
    {
        free(this->data_);
    }
}

bool ArrayNd::is_cuda()
{
    return this->cuda_;
}

void ArrayNd::to_cpu()
{
    if (this->cuda_)
    {
        size_t size = this->size();
        float *dst = (float *)malloc(size);
        cudaMemcpy(dst, this->data_, size, cudaMemcpyDeviceToHost);
        cudaFree(this->data_);
        this->data_ = dst;
        this->cuda_ = false;
    }
}

void ArrayNd::to_cuda()
{
    if (!this->cuda_)
    {
        size_t size = this->size();
        float *dst;
        cudaMalloc(&dst, size);
        cudaMemcpy(dst, this->data_, size, cudaMemcpyHostToDevice);
        free(this->data_);
        this->data_ = dst;
        this->cuda_ = true;
    }
}

float *ArrayNd::data()
{
    return this->data_;
}

int ArrayNd::count()
{
    return this->dims_.size();
}

size_t ArrayNd::size()
{
    return sizeof(float) * this->dims_.size();
}

void ArrayNd::zeros()
{
    size_t size = this->size();

    if (this->cuda_)
    {
        cudaMemset(this->data_, 0, size);
    }
    else
    {
        memset(this->data_, 0, size);
    }
}

void ArrayNd::rands(float mean, float stddev)
{
    bool orig_cuda = this->cuda_;

    this->to_cpu();

    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < this->dims_.size(); i++)
        {
            std::normal_distribution<float> d(mean, stddev);
            this->data_[i] = d(gen);
        }
    }

    if (orig_cuda)
    {
        this->to_cuda();
    }
}

Array1d::Array1d(bool cuda, int cnt)
    : ArrayNd(cuda, Dimensions(cnt))
{
}

Array1d::~Array1d()
{
}

void Array1d::print()
{
    bool orig_cuda = this->cuda_;
    this->to_cpu();

    int cnt = this->cnt();
    printf("[ ");
    for (int i = 0; i < cnt; i++)
    {
        float val = this->data_[i];

        if (i == cnt - 1)
        {
            if (val >= 0.0f)
            {
                printf(" %f", val);
            }
            else
            {
                printf("%f", val);
            }
        }
        else
        {
            if (val >= 0.0f)
            {
                printf(" %f\t", val);
            }
            else
            {
                printf("%f\t", val);
            }
        }
    }
    printf(" ]");

    if (orig_cuda)
    {
        this->to_cuda();
    }
}

int Array1d::cnt()
{
    return this->dims_.dim(0);
}

Array2d::Array2d(bool cuda, int row_cnt, int col_cnt)
    : ArrayNd(cuda, Dimensions(row_cnt, col_cnt))
{
}

Array2d::~Array2d()
{
}

void Array2d::print()
{
    bool orig_cuda = this->cuda_;
    this->to_cpu();

    int row_cnt = this->rows();
    int col_cnt = this->cols();

    printf("[");
    for (int i = 0; i < row_cnt; i++)
    {
        if (i == 0)
        {
            printf(" [ ");
        }
        else
        {
            printf("  [ ");
        }

        for (int j = 0; j < col_cnt; j++)
        {
            float val = this->data_[i * col_cnt + j];

            if (j == col_cnt - 1)
            {
                if (val >= 0.0f)
                {
                    printf(" %f", val);
                }
                else
                {
                    printf("%f", val);
                }
            }
            else
            {
                if (val >= 0.0f)
                {
                    printf(" %f\t", val);
                }
                else
                {
                    printf("%f\t", val);
                }
            }
        }

        if (i == row_cnt - 1)
        {
            printf(" ] ");
        }
        else
        {
            printf(" ],\n");
        }
    }
    printf("]\n");

    if (orig_cuda)
    {
        this->to_cuda();
    }
}

int Array2d::rows()
{
    return this->dims_.dim(0);
}

int Array2d::cols()
{
    return this->dims_.dim(1);
}

Array3d::Array3d(bool cuda, int x_cnt, int y_cnt, int z_cnt)
    : ArrayNd(cuda, Dimensions(x_cnt, y_cnt, z_cnt))
{
}

Array3d::~Array3d()
{
}

void Array3d::print()
{
    bool orig_cuda = this->cuda_;
    this->to_cpu();

    int x_cnt = this->xs();
    int y_cnt = this->ys();
    int z_cnt = this->zs();

    printf("[");
    for (int i = 0; i < x_cnt; i++)
    {
        if (i == 0)
        {
            printf(" [ ");
        }
        else
        {
            printf("  [ ");
        }

        for (int j = 0; j < y_cnt; j++)
        {

            if (j == 0)
            {
                printf(" [ ");
            }
            else
            {
                printf("  [ ");
            }

            for (int k = 0; k < z_cnt; k++)
            {
                float val = this->data_[(i * y_cnt * z_cnt) + (j * z_cnt) + k];

                if (k == z_cnt - 1)
                {
                    if (val >= 0.0f)
                    {
                        printf(" %f", val);
                    }
                    else
                    {
                        printf("%f", val);
                    }
                }
                else
                {
                    if (val >= 0.0f)
                    {
                        printf(" %f\t", val);
                    }
                    else
                    {
                        printf("%f\t", val);
                    }
                }
            }

            if (j == y_cnt - 1)
            {
                printf(" ] ");
            }
            else
            {
                printf(" ],\n");
            }
        }

        if (i == x_cnt - 1)
        {
            printf(" ] ");
        }
        else
        {
            printf(" ],\n");
        }
    }
    printf("]\n");

    if (orig_cuda)
    {
        this->to_cuda();
    }
}

int Array3d::xs()
{
    return this->dims_.dim(0);
}

int Array3d::ys()
{
    return this->dims_.dim(1);
}

int Array3d::zs()
{
    return this->dims_.dim(2);
}

Tensor::Tensor(bool cuda, Dimensions dims)
    : ArrayNd(cuda, dims)
{
}

Tensor::~Tensor()
{
}

void Tensor::print()
{
    bool orig_cuda = this->cuda_;
    this->to_cpu();

    printf("Shape: ");
    this->dims_.print();

    printf("Data: \n");
    for (int i = 0; i < this->dims_.size(); i++)
    {
        printf("%d: %f\n", i, this->data_[i]);
    }

    if (orig_cuda)
    {
        this->to_cuda();
    }
}

Dimensions Tensor::shape()
{
    return this->dims_;
}

int Tensor::num_dims()
{
    return this->dims_.count();
}

int Tensor::dims_size()
{
    return this->dims_.size();
}