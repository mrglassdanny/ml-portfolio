#include "tensor.cuh"

using namespace zero::core;

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

__global__ void k_unpad(float *dst, float *src, int dst_row_cnt, int dst_col_cnt, int src_row_cnt, int src_col_cnt, int pad_row_cnt, int pad_col_cnt)
{
    int dst_col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_col_idx < dst_col_cnt && dst_row_idx < dst_row_cnt)
    {
        dst[dst_row_idx * dst_col_cnt + dst_col_idx] = src[(dst_row_idx + pad_row_cnt) * src_col_cnt + (dst_col_idx + pad_col_cnt)];
    }
}

__global__ void k_sum(float *data, int cnt, float *sum_val)
{
    __shared__ float temp[ZERO_CORE_CUDA_THREADS_PER_BLOCK];
    memset(temp, 0, ZERO_CORE_CUDA_THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        temp[threadIdx.x] = data[tid];
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float l_sum_val = 0.0f;

        for (int i = 0; i < ZERO_CORE_CUDA_THREADS_PER_BLOCK; i++)
        {
            l_sum_val += temp[i];
        }

        atomicAdd(sum_val, l_sum_val);
    }
}

__global__ void k_variance(float *data, int cnt, float mean_val, float *variance_val)
{
    __shared__ float temp[ZERO_CORE_CUDA_THREADS_PER_BLOCK];
    memset(temp, 0, ZERO_CORE_CUDA_THREADS_PER_BLOCK * sizeof(float));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        float diff = data[tid] - mean_val;
        temp[threadIdx.x] = (diff * diff);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        float l_variance_val = 0.0f;

        for (int i = 0; i < ZERO_CORE_CUDA_THREADS_PER_BLOCK; i++)
        {
            l_variance_val += temp[i];
        }

        atomicAdd(variance_val, l_variance_val);
    }
}

__global__ void k_abs(float *data, int cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cnt)
    {
        data[tid] = abs(data[tid]);
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

bool Shape::operator==(const Shape &other)
{
    if (this->dims_.size() != other.dims_.size())
    {
        return false;
    }

    for (int i = 0; i < this->dims_.size(); i++)
    {
        if (this->dims_[i] != other.dims_[i])
        {
            return false;
        }
    }

    return true;
}

bool Shape::operator!=(const Shape &other)
{
    return !(*this == other);
}

void Shape::print()
{
    char buf[64];
    memset(buf, 0, sizeof(buf));

    int cnt = this->num_dims();

    for (int i = 0; i < cnt; i++)
    {
        sprintf(&buf[strlen(buf)], "%d", this->dims_[i]);
        if (i < cnt - 1)
        {
            sprintf(&buf[strlen(buf)], "x");
        }
    }

    printf("%s", buf);
}

void Shape::print_pad(int pad_to_len, bool left_pad)
{
    char buf[64];
    memset(buf, 0, sizeof(buf));

    int cnt = this->num_dims();

    for (int i = 0; i < cnt; i++)
    {
        sprintf(&buf[strlen(buf)], "%d", this->dims_[i]);
        if (i < cnt - 1)
        {
            sprintf(&buf[strlen(buf)], "x");
        }
    }

    if (left_pad)
    {
        char lpad_buf[64];
        memset(lpad_buf, 0, sizeof(lpad_buf));

        for (int i = 0; i < pad_to_len - strlen(buf); i++)
        {
            lpad_buf[i] = ' ';
        }

        printf("%s", lpad_buf);
    }
    else
    {
        for (int i = strlen(buf); i < pad_to_len; i++)
        {
            buf[i] = ' ';
        }
    }

    printf("%s", buf);
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

Tensor::Tensor(Tensor &src)
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

Tensor::Tensor(bool cuda, Shape shape)
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

Tensor::~Tensor()
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

Tensor *Tensor::from_data(Shape shape, float *data)
{
    Tensor *tensor = new Tensor(false, shape);
    cudaMemcpy(tensor->data_, data, tensor->size(), cudaMemcpyDefault);
    return tensor;
}

Tensor *Tensor::from_csv(const char *path)
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

    Tensor *tensor = new Tensor(false, Shape(row_cnt, col_cnt));

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
            tensor->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
            memset(temp_buf, 0, 64);
            col_idx++;
            temp_buf_idx = 0;
        }
        else if (buf[buf_idx] == '\n')
        {
            tensor->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
            memset(temp_buf, 0, 64);
            row_idx++;
            col_idx = 0;
            temp_buf_idx = 0;
        }
    }

    // Make sure to grab the last bit before we finish up!
    if (temp_buf_idx > 0)
    {
        tensor->set_val(row_idx * col_cnt + col_idx, (float)atof(temp_buf));
        memset(temp_buf, 0, 64);
        row_idx++;
        col_idx = 0;
        temp_buf_idx = 0;
    }

    free(buf);

    return tensor;
}

void Tensor::to_csv(const char *path, Tensor *tensor)
{
    int dim_cnt = tensor->num_dims();

    if (dim_cnt == 1)
    {
        int cnt = tensor->shape_[0];

        FILE *file_ptr = fopen(path, "w");

        fprintf(file_ptr, "col\n");

        for (int i = 0; i < cnt; i++)
        {
            fprintf(file_ptr, "%f\n", tensor->get_val(i));
        }

        fclose(file_ptr);
    }
    else if (dim_cnt == 2)
    {

        int row_cnt = tensor->shape_[0];
        int col_cnt = tensor->shape_[1];

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
                    fprintf(file_ptr, "%f,", tensor->get_val(i * col_cnt + j));
                }
                else
                {
                    fprintf(file_ptr, "%f", tensor->get_val(i * col_cnt + j));
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

void Tensor::to_file(const char *path, Tensor *tensor)
{
    bool orig_cuda = tensor->cuda_;

    FILE *file_ptr = fopen(path, "wb");

    tensor->to_cpu();

    fwrite(tensor->data_, sizeof(float), tensor->count(), file_ptr);

    fclose(file_ptr);

    if (orig_cuda)
    {
        tensor->to_cuda();
    }
}

Tensor *Tensor::zeros(bool cuda, Shape shape)
{
    Tensor *tensor = new Tensor(cuda, shape);

    tensor->zeros();

    return tensor;
}

Tensor *Tensor::ones(bool cuda, Shape shape)
{
    Tensor *tensor = new Tensor(cuda, shape);

    tensor->ones();

    return tensor;
}

Tensor *Tensor::full(bool cuda, Shape shape, float val)
{
    Tensor *tensor = new Tensor(cuda, shape);

    tensor->full(val);

    return tensor;
}

Tensor *Tensor::random(bool cuda, Shape shape, float mean, float stddev)
{
    Tensor *tensor = new Tensor(false, shape);
    tensor->random(mean, stddev);

    if (cuda)
    {
        tensor->to_cuda();
    }

    return tensor;
}

Tensor *Tensor::random_ints(bool cuda, Shape shape, int upper_bound)
{
    Tensor *tensor = new Tensor(false, shape);
    tensor->random_ints(upper_bound);

    if (cuda)
    {
        tensor->to_cuda();
    }

    return tensor;
}

Tensor *Tensor::one_hot(Tensor *src, int max_val)
{
    int lst_dim_idx = src->num_dims() - 1;

    if (src->shape()[lst_dim_idx] != 1)
    {
        ZERO_CORE_THROW_ERROR("TENSOR ONE HOT ERROR: last dimension must be 1");
    }

    float min_val = src->min();

    if (min_val < 0.0f)
    {
        ZERO_CORE_THROW_ERROR("TENSOR ONE HOT ERROR: negative numbers not allowed");
    }

    int oh_dim = ((int)max_val) + 1;

    std::vector<int> dst_dims = src->shape().dims();
    dst_dims[lst_dim_idx] = oh_dim;

    Tensor *dst = Tensor::zeros(src->is_cuda(), Shape(dst_dims));

    for (int i = 0; i < src->count(); i++)
    {
        int val = (int)src->get_val(i);
        dst->set_val(i * oh_dim + val, 1.0f);
    }

    return dst;
}

Tensor *Tensor::pad(Tensor *src, int pad_row_cnt, int pad_col_cnt)
{
    if (src->num_dims() < 2)
    {
        ZERO_CORE_THROW_ERROR("TENSOR PAD ERROR: shape must have at least 2 dimensions");
    }

    bool orig_cuda = src->cuda_;
    src->to_cuda();

    int col_dim_idx = src->num_dims() - 1;
    int row_dim_idx = col_dim_idx - 1;

    int src_row_cnt = src->shape()[row_dim_idx];
    int src_col_cnt = src->shape()[col_dim_idx];

    std::vector<int> dst_dims;
    for (int i = 0; i < row_dim_idx; i++)
    {
        dst_dims.push_back(src->shape()[i]);
    }

    int dst_row_cnt = src_row_cnt + (pad_row_cnt * 2);
    int dst_col_cnt = src_col_cnt + (pad_col_cnt * 2);

    dst_dims.push_back(dst_row_cnt);
    dst_dims.push_back(dst_col_cnt);

    Tensor *dst = Tensor::zeros(src->cuda_, Shape(dst_dims));

    int grid_row_cnt = (src_row_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (src_col_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    switch (src->num_dims())
    {
    case 2:
    {
        float *src_data = src->data();
        float *dst_data = dst->data();

        k_pad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                         pad_row_cnt, pad_col_cnt);
    }
    break;
    case 3:
    {
        for (int i = 0; i < src->shape()[0]; i++)
        {
            float *src_data = &src->data()[(i * src_row_cnt * src_col_cnt)];
            float *dst_data = &dst->data()[(i * dst_row_cnt * dst_col_cnt)];

            k_pad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                             pad_row_cnt, pad_col_cnt);
        }
    }
    break;
    case 4:
    {
        for (int i = 0; i < src->shape()[0]; i++)
        {
            for (int j = 0; j < src->shape()[1]; j++)
            {
                float *src_data = &src->data()[(i * src->shape()[1] * src_row_cnt * src_col_cnt) + (j * src_row_cnt * src_col_cnt)];
                float *dst_data = &dst->data()[(i * dst->shape()[1] * dst_row_cnt * dst_col_cnt) + (j * dst_row_cnt * dst_col_cnt)];

                k_pad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                                 pad_row_cnt, pad_col_cnt);
            }
        }
    }
    break;
    default:
        ZERO_CORE_THROW_ERROR("TENSOR PAD ERROR: shape must not have more than 4 dimensions");
        break;
    }

    if (!orig_cuda)
    {
        src->to_cpu();
        dst->to_cpu();
    }

    return dst;
}

Tensor *Tensor::unpad(Tensor *src, int pad_row_cnt, int pad_col_cnt)
{
    if (src->num_dims() < 2)
    {
        ZERO_CORE_THROW_ERROR("TENSOR UNPAD ERROR: shape must have at least 2 dimensions");
    }

    bool orig_cuda = src->cuda_;
    src->to_cuda();

    int col_dim_idx = src->num_dims() - 1;
    int row_dim_idx = col_dim_idx - 1;

    int src_row_cnt = src->shape()[row_dim_idx];
    int src_col_cnt = src->shape()[col_dim_idx];

    std::vector<int> dst_dims;
    for (int i = 0; i < row_dim_idx; i++)
    {
        dst_dims.push_back(src->shape()[i]);
    }

    int dst_row_cnt = src_row_cnt - (pad_row_cnt * 2);
    int dst_col_cnt = src_col_cnt - (pad_col_cnt * 2);

    if (dst_row_cnt < 1)
    {
        ZERO_CORE_THROW_ERROR("TENSOR UNPAD ERROR: padding row count is too large");
    }

    if (dst_col_cnt < 1)
    {
        ZERO_CORE_THROW_ERROR("TENSOR UNPAD ERROR: padding column count is too large");
    }

    dst_dims.push_back(dst_row_cnt);
    dst_dims.push_back(dst_col_cnt);

    Tensor *dst = Tensor::zeros(src->cuda_, Shape(dst_dims));

    int grid_row_cnt = (dst_row_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (dst_col_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    switch (src->num_dims())
    {
    case 2:
    {
        float *src_data = src->data();
        float *dst_data = dst->data();

        k_unpad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                           pad_row_cnt, pad_col_cnt);
    }
    break;
    case 3:
    {
        for (int i = 0; i < src->shape()[0]; i++)
        {
            float *src_data = &src->data()[(i * src_row_cnt * src_col_cnt)];
            float *dst_data = &dst->data()[(i * dst_row_cnt * dst_col_cnt)];

            k_unpad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                               pad_row_cnt, pad_col_cnt);
        }
    }
    break;
    case 4:
    {
        for (int i = 0; i < src->shape()[0]; i++)
        {
            for (int j = 0; j < src->shape()[1]; j++)
            {
                float *src_data = &src->data()[(i * src->shape()[1] * src_row_cnt * src_col_cnt) + (j * src_row_cnt * src_col_cnt)];
                float *dst_data = &dst->data()[(i * dst->shape()[1] * dst_row_cnt * dst_col_cnt) + (j * dst_row_cnt * dst_col_cnt)];

                k_unpad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                                   pad_row_cnt, pad_col_cnt);
            }
        }
    }
    break;
    default:
        ZERO_CORE_THROW_ERROR("TENSOR UNPAD ERROR: shape must not have more than 4 dimensions");
        break;
    }

    if (!orig_cuda)
    {
        src->to_cpu();
        dst->to_cpu();
    }

    return dst;
}

void Tensor::print_vec(float *data, int cnt)
{
    printf("[ ");
    for (int i = 0; i < cnt; i++)
    {
        float val = data[i];

        if (i == cnt - 1)
        {
            if (val >= 0.0f)
            {
                printf(" %f ", val);
            }
            else
            {
                printf("%f ", val);
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

void Tensor::print_mtx(float *data, int row_cnt, int col_cnt, const char *whitespace_str)
{
    printf("%s[\n", whitespace_str);
    for (int i = 0; i < row_cnt; i++)
    {
        printf("%s   ", whitespace_str);

        Tensor::print_vec(&data[i * col_cnt], col_cnt);

        printf("\n");
    }
    printf("%s]\n", whitespace_str);
}

void Tensor::print()
{
    bool orig_cuda = this->cuda_;
    this->to_cpu();

    this->shape_.print();
    printf("\n");

    switch (this->num_dims())
    {
    case 1:
    {
        Tensor::print_vec(this->data_, this->count());
    }
    break;
    case 2:
    {
        Tensor::print_mtx(this->data_, this->shape_[0], this->shape_[1], "");
    }
    break;
    case 3:
    {
        int mtx_cnt = this->shape_[0];
        int row_cnt = this->shape_[1];
        int col_cnt = this->shape_[2];

        printf("[\n");
        for (int i = 0; i < mtx_cnt; i++)
        {
            Tensor::print_mtx(&this->data_[i * row_cnt * col_cnt], row_cnt, col_cnt, "   ");
        }
        printf("]");
    }
    break;
    case 4:
    {
        int mtx_cnt = this->shape_[1];
        int row_cnt = this->shape_[2];
        int col_cnt = this->shape_[3];

        printf("[\n");
        for (int i = 0; i < this->shape_[0]; i++)
        {
            printf("   [\n");
            for (int j = 0; j < mtx_cnt; j++)
            {
                int row_cnt = this->shape_[2];
                int col_cnt = this->shape_[3];

                Tensor::print_mtx(&this->data_[(i * mtx_cnt * row_cnt * col_cnt) + (j * row_cnt * col_cnt)],
                                  row_cnt, col_cnt, "      ");
            }
            printf("   ]\n");
        }
        printf("]");
    }
    break;
    default:
        break;
    }

    printf("\n");

    if (orig_cuda)
    {
        this->to_cuda();
    }
}

void Tensor::copy(Tensor *src)
{
    size_t src_size = src->size();

    if (this->cuda_)
    {
        if (src->cuda_)
        {
            if (this->size() != src_size)
            {
                cudaFree(this->data_);
                cudaMalloc(&this->data_, src_size);
            }

            cudaMemcpy(this->data_, src->data_, src_size, cudaMemcpyDeviceToDevice);
        }
        else
        {
            cudaFree(this->data_);
            this->data_ = (float *)malloc(src->size());
            memcpy(this->data_, src->data_, src_size);
        }
    }
    else
    {
        if (src->cuda_)
        {
            free(this->data_);
            cudaMalloc(&this->data_, src_size);
            cudaMemcpy(this->data_, src->data_, src_size, cudaMemcpyDeviceToDevice);
        }
        else
        {
            if (this->size() != src_size)
            {
                free(this->data_);
                this->data_ = (float *)malloc(src->size());
            }

            memcpy(this->data_, src->data_, src_size);
        }
    }

    this->cuda_ = src->cuda_;
    this->shape_ = src->shape_;
}

void Tensor::reshape(Shape shape)
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

void Tensor::change_dim(int dim_idx, int dim)
{
    std::vector<int> dims = this->shape_.dims();
    dims[dim_idx] = dim;
    this->reshape(Shape(dims));
}

bool Tensor::is_cuda()
{
    return this->cuda_;
}

void Tensor::to_cpu()
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

void Tensor::to_cuda()
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

Shape Tensor::shape()
{
    return this->shape_;
}

int Tensor::num_dims()
{
    return this->shape_.num_dims();
}

int Tensor::dims_size()
{
    return this->shape_.dims_size();
}

size_t Tensor::size()
{
    return sizeof(float) * this->dims_size();
}

int Tensor::count()
{
    return this->dims_size();
}

float Tensor::sum()
{
    float sum_val = 0;

    int cnt = this->count();

    if (this->cuda_)
    {
        Tensor *temp_sum = Tensor::zeros(true, Shape(1));
        k_sum<<<cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data_, cnt, &temp_sum->data()[0]);

        sum_val = temp_sum->get_val(0);
        delete temp_sum;
    }
    else
    {
        for (int i = 0; i < cnt; i++)
        {
            sum_val += this->data_[i];
        }
    }

    return sum_val;
}

float Tensor::min()
{
    float min_val = FLT_MAX;

    float val = 0;

    for (int i = 0; i < this->count(); i++)
    {
        val = this->get_val(i);

        if (val < min_val)
        {
            min_val = val;
        }
    }

    return min_val;
}

int Tensor::min_idx()
{
    float min_val = FLT_MAX;
    int idx;

    float val = 0;

    for (int i = 0; i < this->count(); i++)
    {
        val = this->get_val(i);

        if (val < min_val)
        {
            min_val = val;
            idx = i;
        }
    }

    return idx;
}

float Tensor::max()
{
    float max_val = -FLT_MAX;

    float val = 0;

    for (int i = 0; i < this->count(); i++)
    {
        val = this->get_val(i);

        if (val > max_val)
        {
            max_val = val;
        }
    }

    return max_val;
}

int Tensor::max_idx()
{
    float max_val = -FLT_MAX;
    int idx;

    float val = 0;

    for (int i = 0; i < this->count(); i++)
    {
        val = this->get_val(i);

        if (val > max_val)
        {
            max_val = val;
            idx = i;
        }
    }

    return idx;
}

float Tensor::mean()
{
    return this->sum() / this->count();
}

float Tensor::variance()
{
    float variance_val = 0.0f;

    int cnt = this->count();
    float mean_val = this->mean();

    if (this->cuda_)
    {
        Tensor *temp_variance = Tensor::zeros(true, Shape(1));
        k_variance<<<cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data_, cnt, mean_val, &temp_variance->data()[0]);

        variance_val = temp_variance->get_val(0);
        delete temp_variance;
    }
    else
    {
        for (int i = 0; i < cnt; i++)
        {
            float diff = this->data_[i] - mean_val;
            variance_val += (diff * diff);
        }
    }

    return variance_val /= (float)cnt;
}

float Tensor::stddev()
{
    return sqrt(this->variance());
}

void Tensor::abs()
{
    int cnt = this->count();
    if (this->cuda_)
    {
        k_abs<<<cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data_, cnt);
    }
    else
    {
        for (int i = 0; i < cnt; i++)
        {
            this->data_[i] = fabs(this->data_[i]);
        }
    }
}

float Tensor::get_val(int idx)
{
    float val;
    cudaMemcpy(&val, &this->data_[idx], sizeof(float), cudaMemcpyDefault);
    return val;
}

void Tensor::set_val(int idx, float val)
{
    cudaMemcpy(&this->data_[idx], &val, sizeof(float), cudaMemcpyDefault);
}

float *Tensor::data()
{
    return this->data_;
}

void Tensor::zeros()
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

void Tensor::ones()
{
    if (this->is_cuda())
    {
        k_set_all<<<(this->count() / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1), ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data_, this->count(), 1.0f);
    }
    else
    {
        for (int i = 0; i < this->count(); i++)
        {
            this->data_[i] = 1.0f;
        }
    }
}

void Tensor::full(float val)
{
    if (this->cuda_)
    {
        k_set_all<<<this->count() / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data_, this->count(), val);
    }
    else
    {
        for (int i = 0; i < this->count(); i++)
        {
            this->data_[i] = val;
        }
    }
}

void Tensor::random(float mean, float stddev)
{
    bool orig_cuda = this->cuda_;
    this->to_cpu();

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < this->count(); i++)
    {
        std::normal_distribution<float> d(mean, stddev);
        this->data_[i] = d(gen);
    }

    if (orig_cuda)
    {
        this->to_cuda();
    }
}

void Tensor::random_ints(int upper_bound)
{
    bool orig_cuda = this->cuda_;
    this->to_cpu();

    for (int i = 0; i < this->count(); i++)
    {
        this->data_[i] = rand() % upper_bound;
    }

    if (orig_cuda)
    {
        this->to_cuda();
    }
}
