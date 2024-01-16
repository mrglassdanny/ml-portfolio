#include "autodiff.cuh"

namespace epyon
{
    namespace core
    {
        IntVar::IntVar()
        {
            this->pds[0] = 0.0f;
            this->pds[1] = 0.0f;

            this->ps[0] = nullptr;
            this->ps[1] = nullptr;
        }

        IntVar::IntVar(float pd1, float pd2)
        {
            this->pds[0] = pd1;
            this->pds[1] = pd2;

            this->ps[0] = nullptr;
            this->ps[1] = nullptr;
        }

        IntVar::IntVar(float pd1, float pd2, IntVar *p1, IntVar *p2)
        {
            this->pds[0] = pd1;
            this->pds[1] = pd2;

            this->ps[0] = p1;
            this->ps[1] = p2;
        }

        Var::Var() {}

        Var::Var(float v)
        {
            this->v = v;
            this->iv = nullptr;
        }

        Var::Var(float v, IntVar *iv)
        {
            this->v = v;
            this->iv = iv;
        }

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
            __shared__ float temp[EPYON_CORE_CUDA_THREADS_PER_BLOCK];
            memset(temp, 0, EPYON_CORE_CUDA_THREADS_PER_BLOCK * sizeof(float));

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < cnt)
            {
                temp[threadIdx.x] = data[tid];
            }

            __syncthreads();

            if (threadIdx.x == 0)
            {
                float l_sum_val = 0.0f;

                for (int i = 0; i < EPYON_CORE_CUDA_THREADS_PER_BLOCK; i++)
                {
                    l_sum_val += temp[i];
                }

                atomicAdd(sum_val, l_sum_val);
            }
        }

        __global__ void k_variance(float *data, int cnt, float mean_val, float *variance_val)
        {
            __shared__ float temp[EPYON_CORE_CUDA_THREADS_PER_BLOCK];
            memset(temp, 0, EPYON_CORE_CUDA_THREADS_PER_BLOCK * sizeof(float));

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

                for (int i = 0; i < EPYON_CORE_CUDA_THREADS_PER_BLOCK; i++)
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

        Shape::Shape(int dim_1, Shape shape)
        {
            this->dims.push_back(dim_1);
            for (int dim_i : shape.dims)
            {
                this->dims.push_back(dim_i);
            }
        }

        Shape::~Shape()
        {
        }

        int Shape::operator[](int idx) const
        {
            return this->dims[idx];
        }

        bool Shape::operator==(const Shape &other)
        {
            if (this->dims.size() != other.dims.size())
            {
                return false;
            }

            for (int i = 0; i < this->dims.size(); i++)
            {
                if (this->dims[i] != other.dims[i])
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

            int cnt = this->count();

            for (int i = 0; i < cnt; i++)
            {
                sprintf(&buf[strlen(buf)], "%d", this->dims[i]);
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

            int cnt = this->count();

            for (int i = 0; i < cnt; i++)
            {
                sprintf(&buf[strlen(buf)], "%d", this->dims[i]);
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

        std::vector<int> Shape::get_dims()
        {
            return this->dims;
        }

        int Shape::count()
        {
            return this->dims.size();
        }

        int Shape::size()
        {
            int size = 1;

            for (int i = 0; i < this->count(); i++)
            {
                size *= this->dims[i];
            }

            return size;
        }

        Tensor::Tensor(Tensor &src)
        {
            this->cuda = src.cuda;
            this->shape = src.shape;

            size_t size = this->size();

            if (src.cuda)
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

        Tensor::Tensor(bool cuda, Shape shape)
        {
            this->cuda = cuda;
            this->requires_grad = false;
            this->shape = shape;

            size_t size = this->size();

            if (cuda)
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
            if (this->cuda)
            {
                cudaFree(this->data);
            }
            else
            {
                free(this->data);
            }

            if (this->requires_grad)
            {
                if (this->cuda)
                {
                    cudaFree(this->ivs);
                }
                else
                {
                    free(this->ivs);
                }
            }
        }

        Tensor *Tensor::from_data(Shape shape, float *data)
        {
            Tensor *tensor = new Tensor(false, shape);
            cudaMemcpy(tensor->data, data, tensor->size(), cudaMemcpyDefault);
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
                int cnt = tensor->shape[0];

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

                int row_cnt = tensor->shape[0];
                int col_cnt = tensor->shape[1];

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
            bool orig_cuda = tensor->cuda;

            FILE *file_ptr = fopen(path, "wb");

            tensor->to_cpu();

            fwrite(tensor->data, sizeof(float), tensor->count(), file_ptr);

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

        Tensor *Tensor::fill(bool cuda, Shape shape, float val)
        {
            Tensor *tensor = new Tensor(cuda, shape);

            tensor->fill(val);

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

            if (src->get_shape()[lst_dim_idx] != 1)
            {
                EPYON_CORE_THROW_ERROR("TENSOR ONE HOT ERROR: last dimension must be 1");
            }

            float min_val = src->min();

            if (min_val < 0.0f)
            {
                EPYON_CORE_THROW_ERROR("TENSOR ONE HOT ERROR: negative numbers not allowed");
            }

            int oh_dim = ((int)max_val) + 1;

            std::vector<int> dst_dims = src->get_shape().get_dims();
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
                EPYON_CORE_THROW_ERROR("TENSOR PAD ERROR: shape must have at least 2 dimensions");
            }

            bool orig_cuda = src->cuda;
            src->to_cuda();

            int col_dim_idx = src->num_dims() - 1;
            int row_dim_idx = col_dim_idx - 1;

            int src_row_cnt = src->get_shape()[row_dim_idx];
            int src_col_cnt = src->get_shape()[col_dim_idx];

            std::vector<int> dst_dims;
            for (int i = 0; i < row_dim_idx; i++)
            {
                dst_dims.push_back(src->get_shape()[i]);
            }

            int dst_row_cnt = src_row_cnt + (pad_row_cnt * 2);
            int dst_col_cnt = src_col_cnt + (pad_col_cnt * 2);

            dst_dims.push_back(dst_row_cnt);
            dst_dims.push_back(dst_col_cnt);

            Tensor *dst = Tensor::zeros(src->cuda, Shape(dst_dims));

            int grid_row_cnt = (src_row_cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK) + 1;
            int grid_col_cnt = (src_col_cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK) + 1;

            dim3 grid_dims(grid_col_cnt, grid_row_cnt);
            dim3 block_dims(EPYON_CORE_CUDA_THREADS_PER_BLOCK, EPYON_CORE_CUDA_THREADS_PER_BLOCK);

            switch (src->num_dims())
            {
            case 2:
            {
                float *src_data = src->get_data();
                float *dst_data = dst->get_data();

                k_pad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                                 pad_row_cnt, pad_col_cnt);
            }
            break;
            case 3:
            {
                for (int i = 0; i < src->get_shape()[0]; i++)
                {
                    float *src_data = &src->get_data()[(i * src_row_cnt * src_col_cnt)];
                    float *dst_data = &dst->get_data()[(i * dst_row_cnt * dst_col_cnt)];

                    k_pad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                                     pad_row_cnt, pad_col_cnt);
                }
            }
            break;
            case 4:
            {
                for (int i = 0; i < src->get_shape()[0]; i++)
                {
                    for (int j = 0; j < src->get_shape()[1]; j++)
                    {
                        float *src_data = &src->get_data()[(i * src->get_shape()[1] * src_row_cnt * src_col_cnt) + (j * src_row_cnt * src_col_cnt)];
                        float *dst_data = &dst->get_data()[(i * dst->get_shape()[1] * dst_row_cnt * dst_col_cnt) + (j * dst_row_cnt * dst_col_cnt)];

                        k_pad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                                         pad_row_cnt, pad_col_cnt);
                    }
                }
            }
            break;
            default:
                EPYON_CORE_THROW_ERROR("TENSOR PAD ERROR: shape must not have more than 4 dimensions");
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
                EPYON_CORE_THROW_ERROR("TENSOR UNPAD ERROR: shape must have at least 2 dimensions");
            }

            bool orig_cuda = src->cuda;
            src->to_cuda();

            int col_dim_idx = src->num_dims() - 1;
            int row_dim_idx = col_dim_idx - 1;

            int src_row_cnt = src->get_shape()[row_dim_idx];
            int src_col_cnt = src->get_shape()[col_dim_idx];

            std::vector<int> dst_dims;
            for (int i = 0; i < row_dim_idx; i++)
            {
                dst_dims.push_back(src->get_shape()[i]);
            }

            int dst_row_cnt = src_row_cnt - (pad_row_cnt * 2);
            int dst_col_cnt = src_col_cnt - (pad_col_cnt * 2);

            if (dst_row_cnt < 1)
            {
                EPYON_CORE_THROW_ERROR("TENSOR UNPAD ERROR: padding row count is too large");
            }

            if (dst_col_cnt < 1)
            {
                EPYON_CORE_THROW_ERROR("TENSOR UNPAD ERROR: padding column count is too large");
            }

            dst_dims.push_back(dst_row_cnt);
            dst_dims.push_back(dst_col_cnt);

            Tensor *dst = Tensor::zeros(src->cuda, Shape(dst_dims));

            int grid_row_cnt = (dst_row_cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK) + 1;
            int grid_col_cnt = (dst_col_cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK) + 1;

            dim3 grid_dims(grid_col_cnt, grid_row_cnt);
            dim3 block_dims(EPYON_CORE_CUDA_THREADS_PER_BLOCK, EPYON_CORE_CUDA_THREADS_PER_BLOCK);

            switch (src->num_dims())
            {
            case 2:
            {
                float *src_data = src->get_data();
                float *dst_data = dst->get_data();

                k_unpad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                                   pad_row_cnt, pad_col_cnt);
            }
            break;
            case 3:
            {
                for (int i = 0; i < src->get_shape()[0]; i++)
                {
                    float *src_data = &src->get_data()[(i * src_row_cnt * src_col_cnt)];
                    float *dst_data = &dst->get_data()[(i * dst_row_cnt * dst_col_cnt)];

                    k_unpad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                                       pad_row_cnt, pad_col_cnt);
                }
            }
            break;
            case 4:
            {
                for (int i = 0; i < src->get_shape()[0]; i++)
                {
                    for (int j = 0; j < src->get_shape()[1]; j++)
                    {
                        float *src_data = &src->get_data()[(i * src->get_shape()[1] * src_row_cnt * src_col_cnt) + (j * src_row_cnt * src_col_cnt)];
                        float *dst_data = &dst->get_data()[(i * dst->get_shape()[1] * dst_row_cnt * dst_col_cnt) + (j * dst_row_cnt * dst_col_cnt)];

                        k_unpad<<<grid_dims, block_dims>>>(dst_data, src_data, dst_row_cnt, dst_col_cnt, src_row_cnt, src_col_cnt,
                                                           pad_row_cnt, pad_col_cnt);
                    }
                }
            }
            break;
            default:
                EPYON_CORE_THROW_ERROR("TENSOR UNPAD ERROR: shape must not have more than 4 dimensions");
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
            bool orig_cuda = this->cuda;
            this->to_cpu();

            this->shape.print();
            printf("\n");

            printf("Requires Grad: %s\n", this->requires_grad ? "True" : "False");

            switch (this->num_dims())
            {
            case 1:
            {
                Tensor::print_vec(this->data, this->count());
            }
            break;
            case 2:
            {
                Tensor::print_mtx(this->data, this->shape[0], this->shape[1], "");
            }
            break;
            case 3:
            {
                int mtx_cnt = this->shape[0];
                int row_cnt = this->shape[1];
                int col_cnt = this->shape[2];

                printf("[\n");
                for (int i = 0; i < mtx_cnt; i++)
                {
                    Tensor::print_mtx(&this->data[i * row_cnt * col_cnt], row_cnt, col_cnt, "   ");
                }
                printf("]");
            }
            break;
            case 4:
            {
                int mtx_cnt = this->shape[1];
                int row_cnt = this->shape[2];
                int col_cnt = this->shape[3];

                printf("[\n");
                for (int i = 0; i < this->shape[0]; i++)
                {
                    printf("   [\n");
                    for (int j = 0; j < mtx_cnt; j++)
                    {
                        int row_cnt = this->shape[2];
                        int col_cnt = this->shape[3];

                        Tensor::print_mtx(&this->data[(i * mtx_cnt * row_cnt * col_cnt) + (j * row_cnt * col_cnt)],
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

            if (this->cuda)
            {
                if (src->cuda)
                {
                    if (this->size() != src_size)
                    {
                        cudaFree(this->data);
                        cudaMalloc(&this->data, src_size);
                    }

                    cudaMemcpy(this->data, src->data, src_size, cudaMemcpyDeviceToDevice);
                }
                else
                {
                    cudaFree(this->data);
                    this->data = (float *)malloc(src->size());
                    memcpy(this->data, src->data, src_size);
                }
            }
            else
            {
                if (src->cuda)
                {
                    free(this->data);
                    cudaMalloc(&this->data, src_size);
                    cudaMemcpy(this->data, src->data, src_size, cudaMemcpyDeviceToDevice);
                }
                else
                {
                    if (this->size() != src_size)
                    {
                        free(this->data);
                        this->data = (float *)malloc(src->size());
                    }

                    memcpy(this->data, src->data, src_size);
                }
            }

            this->cuda = src->cuda;
            this->shape = src->shape;
        }

        void Tensor::reshape(Shape shape)
        {
            if (this->shape.size() == shape.size())
            {
                this->shape = shape;
            }
            else
            {
                this->shape = shape;

                if (this->cuda)
                {
                    cudaFree(this->data);
                    cudaMalloc(&this->data, this->size());
                }
                else
                {
                    free(this->data);
                    this->data = (float *)malloc(this->size());
                }
            }
        }

        void Tensor::change_dim(int dim_idx, int dim)
        {
            std::vector<int> dims = this->shape.get_dims();
            dims[dim_idx] = dim;
            this->reshape(Shape(dims));
        }

        bool Tensor::is_cuda()
        {
            return this->cuda;
        }

        void Tensor::to_cpu()
        {
            if (this->cuda)
            {
                size_t size = this->size();
                float *dst = (float *)malloc(size);
                cudaMemcpy(dst, this->data, size, cudaMemcpyDeviceToHost);
                cudaFree(this->data);
                this->data = dst;
                this->cuda = false;
            }
        }

        void Tensor::to_cuda()
        {
            if (!this->cuda)
            {
                size_t size = this->size();
                float *dst;
                cudaMalloc(&dst, size);
                cudaMemcpy(dst, this->data, size, cudaMemcpyHostToDevice);
                free(this->data);
                this->data = dst;
                this->cuda = true;
            }
        }

        void Tensor::require_grad(Context *ctx)
        {
            if (this->requires_grad)
                return;

            this->requires_grad = true;

            if (this->cuda)
            {
                cudaMalloc(&this->ivs, sizeof(IntVar *) * this->count());
            }
            else
            {
                this->ivs = (IntVar **)malloc(sizeof(IntVar *) * this->count());
            }

            for (int i = 0; i < this->count(); i++)
            {
                this->ivs[i] = ctx->add_intvar(IntVar());
            }
        }

        Shape Tensor::get_shape()
        {
            return this->shape;
        }

        int Tensor::num_dims()
        {
            return this->shape.count();
        }

        int Tensor::count()
        {
            return this->shape.size();
        }

        size_t Tensor::size()
        {
            return sizeof(float) * this->count();
        }

        float Tensor::sum()
        {
            float sum_val = 0;

            int cnt = this->count();

            if (this->cuda)
            {
                Tensor *temp_sum = Tensor::zeros(true, Shape(1));
                k_sum<<<cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK + 1, EPYON_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data, cnt, &temp_sum->get_data()[0]);

                sum_val = temp_sum->get_val(0);
                delete temp_sum;
            }
            else
            {
                for (int i = 0; i < cnt; i++)
                {
                    sum_val += this->data[i];
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

            if (this->cuda)
            {
                Tensor *temp_variance = Tensor::zeros(true, Shape(1));
                k_variance<<<cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK + 1, EPYON_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data, cnt, mean_val, &temp_variance->get_data()[0]);

                variance_val = temp_variance->get_val(0);
                delete temp_variance;
            }
            else
            {
                for (int i = 0; i < cnt; i++)
                {
                    float diff = this->data[i] - mean_val;
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
            if (this->cuda)
            {
                k_abs<<<cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK + 1, EPYON_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data, cnt);
            }
            else
            {
                for (int i = 0; i < cnt; i++)
                {
                    this->data[i] = fabs(this->data[i]);
                }
            }
        }

        float Tensor::get_val(int idx)
        {
            float val;
            cudaMemcpy(&val, &this->data[idx], sizeof(float), cudaMemcpyDefault);
            return val;
        }

        void Tensor::set_val(int idx, float val)
        {
            cudaMemcpy(&this->data[idx], &val, sizeof(float), cudaMemcpyDefault);
        }

        float *Tensor::get_data()
        {
            return this->data;
        }

        void Tensor::zeros()
        {
            size_t size = this->size();

            if (this->cuda)
            {
                cudaMemset(this->data, 0, size);
            }
            else
            {
                memset(this->data, 0, size);
            }
        }

        void Tensor::ones()
        {
            if (this->is_cuda())
            {
                k_set_all<<<(this->count() / EPYON_CORE_CUDA_THREADS_PER_BLOCK + 1), EPYON_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data, this->count(), 1.0f);
            }
            else
            {
                for (int i = 0; i < this->count(); i++)
                {
                    this->data[i] = 1.0f;
                }
            }
        }

        void Tensor::fill(float val)
        {
            if (this->cuda)
            {
                k_set_all<<<this->count() / EPYON_CORE_CUDA_THREADS_PER_BLOCK + 1, EPYON_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data, this->count(), val);
            }
            else
            {
                for (int i = 0; i < this->count(); i++)
                {
                    this->data[i] = val;
                }
            }
        }

        void Tensor::random(float mean, float stddev)
        {
            bool orig_cuda = this->cuda;
            this->to_cpu();

            std::random_device rd;
            std::mt19937 gen(rd());

            for (int i = 0; i < this->count(); i++)
            {
                std::normal_distribution<float> d(mean, stddev);
                this->data[i] = d(gen);
            }

            if (orig_cuda)
            {
                this->to_cuda();
            }
        }

        void Tensor::random_ints(int upper_bound)
        {
            bool orig_cuda = this->cuda;
            this->to_cpu();

            for (int i = 0; i < this->count(); i++)
            {
                this->data[i] = rand() % upper_bound;
            }

            if (orig_cuda)
            {
                this->to_cuda();
            }
        }

        Context::Context(bool cuda)
        {
            this->cuda = cuda;

            if (this->cuda)
            {
                cudaMalloc(&this->tape_blocks, sizeof(IntVar *) * EPYON_AD_TAPE_BLOCK_CNT);
            }
            else
            {
                this->tape_blocks = (IntVar **)malloc(sizeof(IntVar *) * EPYON_AD_TAPE_BLOCK_CNT);
            }

            this->tape_block_cur = -1;
            this->tape_iv_cur = -1;

            this->add_block();
        }

        Context::~Context()
        {
            if (this->cuda)
            {
                for (int i = 0; i < this->tape_block_cur + 1; i++)
                {
                    cudaFree(this->tape_blocks[i]);
                }
                cudaFree(this->tape_blocks);
            }
            else
            {
                for (int i = 0; i < this->tape_block_cur + 1; i++)
                {
                    free(this->tape_blocks[i]);
                }
                free(this->tape_blocks);
            }
        }

        __host__ __device__ void Context::add_block()
        {
            if (this->cuda)
            {
                cudaMalloc(&this->tape_blocks[++this->tape_block_cur], sizeof(IntVar) * EPYON_AD_TAPE_BLOCK_SIZE);
            }
            else
            {
                this->tape_blocks[++this->tape_block_cur] = (IntVar *)malloc(sizeof(IntVar) * EPYON_AD_TAPE_BLOCK_SIZE);
            }

            this->tape_iv_cur = -1;
        }

        __host__ __device__ IntVar *Context::add_intvar(IntVar iv)
        {
            if (this->tape_iv_cur != 0 && this->tape_iv_cur >= EPYON_AD_TAPE_BLOCK_SIZE)
            {
                this->add_block();
            }

            this->tape_blocks[this->tape_block_cur][++this->tape_iv_cur] = iv;
            return &this->tape_blocks[this->tape_block_cur][this->tape_iv_cur];
        }

        __host__ __device__ Var Context::op(float v, float pd1, float pd2, IntVar *p1, IntVar *p2)
        {
            Var var(v);
            var.iv = this->add_intvar(IntVar(pd1, pd2, p1, p2));
            return var;
        }

        __host__ __device__ Var Context::var(float v)
        {
            Var var(v);
            var.iv = this->add_intvar(IntVar());
            return v;
        }

        __host__ __device__ void Context::backward()
        {
            this->tape_blocks[this->tape_block_cur][this->tape_iv_cur].d = 1.0f;

            int tape_iv_cur = this->tape_iv_cur;
            for (int i = this->tape_block_cur; i >= 0; i--)
            {
                for (int j = tape_iv_cur; j >= 0; j--)
                {
                    if (this->tape_blocks[i][j].ps[0] != nullptr)
                        this->tape_blocks[i][j].ps[0]->d += this->tape_blocks[i][j].pds[0] * this->tape_blocks[i][j].d;
                    if (this->tape_blocks[i][j].ps[1] != nullptr)
                        this->tape_blocks[i][j].ps[1]->d += this->tape_blocks[i][j].pds[1] * this->tape_blocks[i][j].d;
                }
                tape_iv_cur = EPYON_AD_TAPE_BLOCK_SIZE;
            }
        }

        __host__ __device__ Var Context::add(Var a, Var b)
        {
            return this->op(a.v + b.v,
                            1.0f, 1.0f,
                            a.iv, b.iv);
        }

        __host__ __device__ Var Context::mul(Var a, Var b)
        {
            return this->op(a.v * b.v,
                            b.v, a.v,
                            a.iv, b.iv);
        }

        __host__ __device__ Var Context::exp(Var a, float b)
        {
            return this->op(pow(a.v, b),
                            b * pow(a.v, b - 1), 0.0f,
                            a.iv, nullptr);
        }
    }
}