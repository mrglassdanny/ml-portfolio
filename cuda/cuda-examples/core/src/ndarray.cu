#include "ndarray.cuh"

__global__ void k_set_all(float *data, int cnt, float val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        data[tid] = val;
    }
}

Shape::Shape()
{
}

Shape::Shape(int dim_1)
{
    this->dims_.push_back(dim_1);
}

Shape::Shape(int dim_1, int dim_2)
{
    this->dims_.push_back(dim_1);
    this->dims_.push_back(dim_2);
}

Shape::Shape(int dim_1, int dim_2, int dim_3)
{
    this->dims_.push_back(dim_1);
    this->dims_.push_back(dim_2);
    this->dims_.push_back(dim_3);
}

Shape::Shape(int dim_1, int dim_2, int dim_3, int dim_4)
{
    this->dims_.push_back(dim_1);
    this->dims_.push_back(dim_2);
    this->dims_.push_back(dim_3);
    this->dims_.push_back(dim_4);
}

Shape::Shape(std::vector<int> dims)
{
    this->dims_ = dims;
}

Shape::~Shape()
{
}

void Shape::print()
{
    int cnt = this->count();

    for (int i = 0; i < cnt; i++)
    {
        printf("%d", this->dims_[i]);

        if (i < cnt - 1)
        {
            printf("x");
        }
    }

    printf("\n");
}

int Shape::dim(int dim_idx)
{
    return this->dims_[dim_idx];
}

std::vector<int> Shape::dims()
{
    return this->dims_;
}

int Shape::count()
{
    return this->dims_.size();
}

int Shape::size()
{
    int size = 1;

    for (int i = 0; i < this->count(); i++)
    {
        size *= this->dims_[i];
    }

    return size;
}

NdArray::NdArray(NdArray &src)
{
    this->cuda_ = src.cuda_;
    this->shape_ = src.shape_;

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

NdArray::NdArray(bool cuda, Shape shape)
{
    this->cuda_ = cuda;
    this->shape_ = shape;

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

NdArray::NdArray(bool cuda, int cnt)
    : NdArray(cuda, Shape(cnt))
{
}

NdArray::NdArray(bool cuda, int row_cnt, int col_cnt)
    : NdArray(cuda, Shape(row_cnt, col_cnt))
{
}

NdArray::NdArray(bool cuda, int x_cnt, int y_cnt, int z_cnt)
    : NdArray(cuda, Shape(x_cnt, y_cnt, z_cnt))
{
}

NdArray::~NdArray()
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

NdArray *NdArray::zeros(bool cuda, Shape shape)
{
    NdArray *arr = new NdArray(cuda, shape);

    arr->zeros();

    return arr;
}

NdArray *NdArray::ones(bool cuda, Shape shape)
{
    NdArray *arr = new NdArray(cuda, shape);

    arr->ones();

    return arr;
}

NdArray *NdArray::rands(bool cuda, Shape shape, float mean, float stddev)
{
    NdArray *arr = new NdArray(cuda, shape);

    arr->rands(mean, stddev);

    return arr;
}

void NdArray::print()
{
    bool orig_cuda = this->cuda_;
    this->to_cpu();

    switch (this->num_dims())
    {
    case 1:
    {
        int cnt = this->count();
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
    }
    break;
    case 2:
    {
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
    }
    break;
    case 3:
    {

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
    }
    break;
    default:
    {
        printf("Shape: ");
        this->shape_.print();

        printf("Data: \n");
        for (int i = 0; i < this->shape_.size(); i++)
        {
            printf("%d: %f\n", i, this->data_[i]);
        }
    }
    break;
    }

    if (orig_cuda)
    {
        this->to_cuda();
    }
}

void NdArray::copy(NdArray *src)
{
    this->cuda_ = src->cuda_;
    this->shape_ = src->shape_;
    cudaMemcpy(this->data_, src->data_, src->size(), cudaMemcpyDefault);
}

void NdArray::reshape(Shape shape)
{
    this->shape_ = shape;

    if (this->cuda_)
    {
        cudaFree(this->data_);
        cudaMalloc(&this->data_, this->size());
    }
    else
    {
        free(this->data_);
        this->data_ = (float *)malloc(this->size());
    }
}

void NdArray::change_dim(int dim_idx, int dim)
{
    std::vector<int> dims = this->shape_.dims();
    dims[dim_idx] = dim;
    this->reshape(Shape(dims));
}

bool NdArray::is_cuda()
{
    return this->cuda_;
}

void NdArray::to_cpu()
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

void NdArray::to_cuda()
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

Shape NdArray::shape()
{
    return this->shape_;
}

int NdArray::num_dims()
{
    return this->shape_.count();
}

int NdArray::dims_size()
{
    return this->shape_.size();
}

int NdArray::count()
{
    return this->shape_.size();
}

int NdArray::rows()
{
    return this->shape_.dim(0);
}

int NdArray::cols()
{
    return this->shape_.dim(1);
}

int NdArray::xs()
{
    return this->shape_.dim(0);
}

int NdArray::ys()
{
    return this->shape_.dim(1);
}

int NdArray::zs()
{
    return this->shape_.dim(2);
}

size_t NdArray::size()
{
    return sizeof(float) * this->shape_.size();
}

float *NdArray::data()
{
    return this->data_;
}

void NdArray::zeros()
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

void NdArray::ones()
{
    if (this->is_cuda())
    {
        k_set_all<<<(this->count() / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK>>>(this->data_, this->count(), 1.0f);
    }
    else
    {
        for (int i = 0; i < this->count(); i++)
        {
            this->data_[i] = 1.0f;
        }
    }
}

void NdArray::rands(float mean, float stddev)
{
    bool orig_cuda = this->cuda_;

    this->to_cpu();

    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < this->shape_.size(); i++)
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

float NdArray::get_val(int idx)
{
    float val;
    cudaMemcpy(&val, &this->data_[idx], sizeof(float), cudaMemcpyDefault);
    return val;
}

void NdArray::set_val(int idx, float val)
{
    cudaMemcpy(&this->data_[idx], &val, sizeof(float), cudaMemcpyDefault);
}

float NdArray::sum()
{
    float sum = 0.0f;

    for (int i = 0; i < this->count(); i++)
    {
        sum += this->get_val(i);
    }

    return sum;
}

float NdArray::min()
{
    float min = FLT_MAX;

    float val = 0;

    for (int i = 0; i < this->count(); i++)
    {
        val = this->get_val(i);

        if (val < min)
        {
            min = val;
        }
    }

    return min;
}

float NdArray::max()
{
    float max = -FLT_MAX;

    float val = 0;

    for (int i = 0; i < this->count(); i++)
    {
        val = this->get_val(i);

        if (val > max)
        {
            max = val;
        }
    }

    return max;
}

float NdArray::mean()
{
    return this->sum() / this->count();
}

float NdArray::stddev()
{
    float stddev = 0.0f;

    float mean = this->mean();

    for (int i = 0; i < this->count(); i++)
    {
        float diff = this->get_val(i) - mean;
        stddev == diff *diff;
    }

    stddev /= this->count();

    return sqrt(stddev);
}