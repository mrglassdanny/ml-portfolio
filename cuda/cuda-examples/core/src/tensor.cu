#include "tensor.cuh"

Dimensions::Dimensions()
{
}

Dimensions::Dimensions(int dim_1)
{
    this->dim_list.push_back(dim_1);
}

Dimensions::Dimensions(int dim_1, int dim_2)
{
    this->dim_list.push_back(dim_1);
    this->dim_list.push_back(dim_2);
}

Dimensions::Dimensions(int dim_1, int dim_2, int dim_3)
{
    this->dim_list.push_back(dim_1);
    this->dim_list.push_back(dim_2);
    this->dim_list.push_back(dim_3);
}

Dimensions::Dimensions(int dim_1, int dim_2, int dim_3, int dim_4)
{
    this->dim_list.push_back(dim_1);
    this->dim_list.push_back(dim_2);
    this->dim_list.push_back(dim_3);
    this->dim_list.push_back(dim_4);
}

Dimensions::Dimensions(std::vector<int> dim_list)
{
    this->dim_list = dim_list;
}

Dimensions::~Dimensions()
{
}

void Dimensions::print()
{
    int dims_cnt = this->get_cnt();

    for (int i = 0; i < dims_cnt; i++)
    {
        printf("%d", this->dim_list[i]);

        if (i < dims_cnt - 1)
        {
            printf("x");
        }
    }

    printf("\n");
}

int Dimensions::get_dim(int dim_idx)
{
    return this->dim_list[dim_idx];
}

int Dimensions::get_cnt()
{
    return this->dim_list.size();
}

int Dimensions::get_size()
{
    int dims_size = 1;

    for (int i = 0; i < this->get_cnt(); i++)
    {
        dims_size *= this->dim_list[i];
    }

    return dims_size;
}

ArrayNd::ArrayNd(ArrayNd &src)
{
    this->cuda_flg = src.cuda_flg;
    this->dims = src.dims;

    size_t size = this->get_data_size();

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

ArrayNd::ArrayNd(bool cuda_flg, Dimensions dims)
{
    this->cuda_flg = cuda_flg;
    this->dims = dims;

    size_t size = this->get_data_size();

    if (cuda_flg)
    {
        cudaMalloc(&this->data, size);
    }
    else
    {
        this->data = (float *)malloc(size);
    }
}

ArrayNd::~ArrayNd()
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

size_t ArrayNd::get_data_size()
{
    return sizeof(float) * this->dims.get_size();
}

bool ArrayNd::is_cuda()
{
    return this->cuda_flg;
}

float *ArrayNd::get_data()
{
    return this->data;
}

void ArrayNd::to_cpu()
{
    if (this->cuda_flg)
    {
        size_t size = this->get_data_size();
        float *dst = (float *)malloc(size);
        cudaMemcpy(dst, this->data, size, cudaMemcpyDeviceToHost);
        cudaFree(this->data);
        this->data = dst;
        this->cuda_flg = false;
    }
}

void ArrayNd::to_cuda()
{
    if (!this->cuda_flg)
    {
        size_t size = this->get_data_size();
        float *dst;
        cudaMalloc(&dst, size);
        cudaMemcpy(dst, this->data, size, cudaMemcpyHostToDevice);
        free(this->data);
        this->data = dst;
        this->cuda_flg = true;
    }
}

void ArrayNd::zeros()
{
    size_t size = this->get_data_size();

    if (this->cuda_flg)
    {
        cudaMemset(this->data, 0, size);
    }
    else
    {
        memset(this->data, 0, size);
    }
}

void ArrayNd::rands(float mean, float stddev)
{
    bool orig_cuda_flg = this->cuda_flg;

    this->to_cpu();

    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < this->dims.get_size(); i++)
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

Array1d::Array1d(bool cuda_flg, int cnt)
    : ArrayNd(cuda_flg, Dimensions(cnt))
{

}

Array1d::~Array1d()
{

}

void Array1d::print()
{
    bool orig_cuda_flg = this->cuda_flg;
    this->to_cpu();

    int cnt = this->get_cnt();
        printf("[ ");
        for (int i = 0; i < cnt; i++)
        {
            float val = this->data[i];

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

        if (orig_cuda_flg)
        {
            this->to_cuda();
        }
    
}

int Array1d::get_cnt()
{
    return this->dims.get_dim(0);
}

Array2d::Array2d(bool cuda_flg, int row_cnt, int col_cnt)
    : ArrayNd(cuda_flg, Dimensions(row_cnt, col_cnt))
{

}

Array2d::~Array2d()
{

}

void Array2d::print()
{
    bool orig_cuda_flg = this->cuda_flg;
    this->to_cpu();

    int row_cnt = this->get_row_cnt();
    int col_cnt = this->get_col_cnt();

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
                float val = this->data[i * col_cnt + j];

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

        if (orig_cuda_flg)
        {
            this->to_cuda();
        }
}

int Array2d::get_row_cnt()
{
    return this->dims.get_dim(0);
}

int Array2d::get_col_cnt()
{
    return this->dims.get_dim(1);
}

Array3d::Array3d(bool cuda_flg, int x, int y, int z)
    : ArrayNd(cuda_flg, Dimensions(x, y, z))
{

}

Array3d::~Array3d()
{

}

void Array3d::print()
{
    bool orig_cuda_flg = this->cuda_flg;
    this->to_cpu();

    int x = this->get_x();
    int y = this->get_y();
    int z = this->get_z();

    printf("[");
    for (int i = 0; i < x; i++)
    {

        if (i == 0)
        {
            printf(" [ ");
        }
        else
        {
            printf("  [ ");
        }

        for (int j = 0; j < y; j++)
        {

            if (j == 0)
            {
                printf(" [ ");
            }
            else
            {
                printf("  [ ");
            }

            for (int k = 0; k < z; k++)
            {
                float val = this->data[(i * y * z) + (j * z) + k];

                if (k == z - 1)
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

            if (j == y - 1)
            {
                printf(" ] ");
            }
            else
            {
                printf(" ],\n");
            }
        }

        if (i == x - 1)
        {
            printf(" ] ");
        }
        else
        {
            printf(" ],\n");
        }
    }
    printf("]\n");

    if (orig_cuda_flg)
    {
        this->to_cuda();
    }
}

int Array3d::get_x()
{
    return this->dims.get_dim(0);
}

int Array3d::get_y()
{
    return this->dims.get_dim(1);
}

int Array3d::get_z()
{
    return this->dims.get_dim(2);
}

Tensor::Tensor(bool cuda_flg, Dimensions dims)
    : ArrayNd(cuda_flg, dims)
{

}

Tensor::~Tensor()
{

}

void Tensor::print()
{
    bool orig_cuda_flg = this->cuda_flg;
    this->to_cpu();

    printf("Shape: ");
    this->dims.print();

    printf("Data: \n");
    for (int i = 0; i < this->dims.get_size(); i++)
    {
        printf("%d: %f\n", i, this->data[i]);
    }

    if (orig_cuda_flg)
    {
        this->to_cuda();
    }
}

Dimensions Tensor::get_dims()
{
    return this->dims;
}

int Tensor::num_dims()
{
    return this->dims.get_cnt();
}

int Tensor::dims_size()
{
    return this->dims.get_size();
}