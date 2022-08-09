#include "ndarray.cuh"

__global__ void k_set_all(float *data, int cnt, float val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        data[tid] = val;
    }
}

__global__ void k_pad(float *dst, float *src, int dst_row_cnt, int dst_col_cnt, int src_row_cnt, int src_col_cnt, int pad_row_cnt, int pad_col_cnt)
{
    int src_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (src_col_idx < src_col_cnt && src_row_idx < src_row_cnt)
    {
        dst[(src_row_idx + pad_row_cnt) * dst_col_cnt + (src_col_idx + pad_col_cnt)] = src[src_row_idx * src_col_cnt + src_col_idx];
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

Shape::Shape(int dim_1, Shape shape)
{
    this->dims_.push_back(dim_1);
    for (int dim_i : shape.dims_)
    {
        this->dims_.push_back(dim_i);
    }
}

Shape::~Shape()
{
}

int Shape::operator[](int idx) const
{
    return this->dims_[idx];
}

void Shape::print()
{
    int cnt = this->num_dims();

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

std::vector<int> Shape::dims()
{
    return this->dims_;
}

int Shape::num_dims()
{
    return this->dims_.size();
}

int Shape::dims_size()
{
    int size = 1;

    for (int i = 0; i < this->num_dims(); i++)
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

NdArray *NdArray::from_csv(const char *path)
{
    FILE *file_ptr = fopen(path, "rb");

    fseek(file_ptr, 0L, SEEK_END);
    long long file_size = FileUtils::get_file_size(path);
    rewind(file_ptr);

    char *buf = (char *)malloc(file_size + 1);
    memset(buf, 0, file_size + 1);
    fread(buf, 1, file_size, file_ptr);

    fclose(file_ptr);

    int buf_idx = 0;

    int row_cnt = 0;
    int col_cnt = 0;

    while (buf[buf_idx] != '\n')
    {
        if (buf[buf_idx] == ',')
        {
            col_cnt++;
        }

        buf_idx++;
    }

    col_cnt++;
    buf_idx++;

    int lst_row_idx = 0;
    for (int i = buf_idx; i < file_size; i++)
    {
        if (buf[i] == '\n')
        {
            row_cnt++;
            lst_row_idx = i;
        }
    }

    // If file does not end in newline, add to the row count.
    if (lst_row_idx < file_size - 1)
    {
        row_cnt++;
    }

    NdArray *ndarray = new NdArray(false, Shape(row_cnt, col_cnt));

    char temp_buf[64];
    memset(temp_buf, 0, 64);
    int temp_buf_idx = 0;
    int row_idx = 0;
    int col_idx = 0;

    for (; buf_idx < file_size; buf_idx++)
    {
        while (buf[buf_idx] != ',' && buf[buf_idx] != '\n' && buf_idx < file_size)
        {
            if (buf[buf_idx] != '"')
            {
                temp_buf[temp_buf_idx++] = buf[buf_idx];
            }

            buf_idx++;
        }

        if (buf[buf_idx] == ',')
        {
            ndarray->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
            memset(temp_buf, 0, 64);
            col_idx++;
            temp_buf_idx = 0;
        }
        else if (buf[buf_idx] == '\n')
        {
            ndarray->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
            memset(temp_buf, 0, 64);
            row_idx++;
            col_idx = 0;
            temp_buf_idx = 0;
        }
    }

    // Make sure to grab the last bit before we finish up!
    if (temp_buf_idx > 0)
    {
        ndarray->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
        memset(temp_buf, 0, 64);
        row_idx++;
        col_idx = 0;
        temp_buf_idx = 0;
    }

    free(buf);

    return ndarray;
}

void NdArray::to_csv(const char *path, NdArray *ndarray)
{
    int dim_cnt = ndarray->num_dims();

    if (dim_cnt == 1)
    {
        int cnt = ndarray->shape_[0];

        FILE *file_ptr = fopen(path, "w");

        fprintf(file_ptr, "col\n");

        for (int i = 0; i < cnt; i++)
        {
            fprintf(file_ptr, "%f\n", ndarray->get_val(i));
        }

        fclose(file_ptr);
    }
    else if (dim_cnt == 2)
    {

        int row_cnt = ndarray->shape_[0];
        int col_cnt = ndarray->shape_[1];

        FILE *file_ptr = fopen(path, "w");

        for (int j = 0; j < col_cnt; j++)
        {

            if (j < col_cnt - 1)
            {
                fprintf(file_ptr, "col_%d,", j);
            }
            else
            {
                fprintf(file_ptr, "col_%d", j);
            }
        }
        fprintf(file_ptr, "\n");

        for (int i = 0; i < row_cnt; i++)
        {
            for (int j = 0; j < col_cnt; j++)
            {
                if (j < col_cnt - 1)
                {
                    fprintf(file_ptr, "%f,", ndarray->get_val(i * col_cnt + j));
                }
                else
                {
                    fprintf(file_ptr, "%f", ndarray->get_val(i * col_cnt + j));
                }
            }
            fprintf(file_ptr, "\n");
        }
        fclose(file_ptr);
    }
    else
    {
        return;
    }
}

void NdArray::to_file(const char *path, NdArray *ndarray)
{
    bool orig_cuda = ndarray->cuda_;

    FILE *file_ptr = fopen(path, "wb");

    ndarray->to_cpu();

    fwrite(ndarray->data_, sizeof(float), ndarray->count(), file_ptr);

    fclose(file_ptr);

    if (orig_cuda)
    {
        ndarray->to_cuda();
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

NdArray *NdArray::full(bool cuda, Shape shape, float val)
{
    NdArray *arr = new NdArray(cuda, shape);

    arr->full(val);

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

    printf("Shape: ");
    this->shape_.print();

    for (int i = 0; i < this->num_dims(); i++)
    {
    }

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
        int row_cnt = this->shape_[0];
        int col_cnt = this->shape_[1];

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

        int x_cnt = this->shape_[0];
        int y_cnt = this->shape_[1];
        int z_cnt = this->shape_[2];

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
    case 4:
    {
        for (int _i = 0; _i < this->shape_[0]; _i++)
        {
            for (int _j = 0; _j < this->shape_[1]; _j++)
            {
                int row_cnt = this->shape_[2];
                int col_cnt = this->shape_[3];

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
                        float val = this->data_[(_i * this->shape_[1] * row_cnt * col_cnt) + (_j * row_cnt * col_cnt) + (i * col_cnt + j)];

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
        }
    }
    break;
    default:
    {

        printf("Data: \n");
        for (int i = 0; i < this->shape_.dims_size(); i++)
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
    return this->shape_.num_dims();
}

int NdArray::dims_size()
{
    return this->shape_.dims_size();
}

int NdArray::count()
{
    return this->shape_.dims_size();
}

size_t NdArray::size()
{
    return sizeof(float) * this->shape_.dims_size();
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

void NdArray::full(float val)
{
    if (this->cuda_)
    {
        k_set_all<<<this->count() / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >>>(this->data_, this->count(), val);
    }
    else
    {
        for (int i = 0; i < this->count(); i++)
        {
            this->data_[i] = val;
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

        for (int i = 0; i < this->shape_.dims_size(); i++)
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

void NdArray::pad(int pad_row_cnt, int pad_col_cnt)
{
    NdArray *prev = new NdArray(*this);

    this->change_dim(0, pad_row_cnt * 2 + prev->shape_[0]);
    this->change_dim(1, pad_col_cnt * 2 + prev->shape_[1]);

    this->zeros();

    if (this->cuda_)
    {
        int grid_row_cnt = (prev->shape_[0] / THREADS_PER_BLOCK) + 1;
        int grid_col_cnt = (prev->shape_[1] / THREADS_PER_BLOCK) + 1;

        dim3 grid_dims(grid_col_cnt, grid_row_cnt);
        dim3 block_dims(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

        k_pad<<<grid_dims, block_dims>>>(this->data(), prev->data(), this->shape_[0], this->shape_[1], prev->shape_[0], prev->shape_[1],
                                         pad_row_cnt, pad_col_cnt);
    }
    else
    {
        for (int i = 0; i < prev->shape_[0]; i++)
        {
            for (int j = 0; j < prev->shape_[1]; j++)
            {
                this->set_val((i + pad_row_cnt) * this->shape_[1] + (j + pad_col_cnt), prev->get_val(i * prev->shape_[1] + j));
            }
        }
    }

    delete prev;
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
        stddev = diff * diff;
    }

    stddev /= this->count();

    return sqrt(stddev);
}
