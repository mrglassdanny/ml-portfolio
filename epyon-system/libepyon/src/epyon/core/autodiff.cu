#include "autodiff.cuh"

namespace epyon
{
    namespace core
    {
        __global__ void k_set_all(Var *data, int cnt, float val)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < cnt)
            {
                data[tid].v = val;
            }
        }

        __global__ void k_sum(AutoDiffContext *ctx, Var *data, int cnt, Var *sum_var)
        {
            __shared__ Var temp[EPYON_CORE_CUDA_THREADS_PER_BLOCK];
            memset(temp, 0, EPYON_CORE_CUDA_THREADS_PER_BLOCK * sizeof(Var));

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < cnt)
            {
                temp[threadIdx.x] = data[tid];
            }

            __syncthreads();

            if (threadIdx.x == 0)
            {
                Var l_sum_var(0.0f);

                for (int i = 0; i < EPYON_CORE_CUDA_THREADS_PER_BLOCK; i++)
                {
                    l_sum_var = ctx->add(temp[i], l_sum_var);
                }

                atomicAdd(&sum_var->v, l_sum_var.v);
            }
        }

        __host__ __device__ IntVar::IntVar()
        {
            this->pds[0] = 0.0f;
            this->pds[1] = 0.0f;

            this->ps[0] = EPYON_AD_DEFAULT_TAPE_INDEX;
            this->ps[1] = EPYON_AD_DEFAULT_TAPE_INDEX;
        }

        __host__ __device__ IntVar::IntVar(float pd1, float pd2)
        {
            this->pds[0] = pd1;
            this->pds[1] = pd2;

            this->ps[0] = EPYON_AD_DEFAULT_TAPE_INDEX;
            this->ps[1] = EPYON_AD_DEFAULT_TAPE_INDEX;
        }

        __host__ __device__ IntVar::IntVar(float pd1, float pd2, TapeIndex p1, TapeIndex p2)
        {
            this->pds[0] = pd1;
            this->pds[1] = pd2;

            this->ps[0] = p1;
            this->ps[1] = p2;
        }

        __host__ __device__ Var::Var() {}

        __host__ __device__ Var::Var(float v)
        {
            this->v = v;
            this->i = EPYON_AD_DEFAULT_TAPE_INDEX;
        }

        __host__ __device__ Var::Var(float v, TapeIndex i)
        {
            this->v = v;
            this->i = i;
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

        Tensor::Tensor(bool cuda, Shape shape)
        {
            this->cuda = cuda;
            this->shape = shape;

            size_t size = this->size();

            if (cuda)
            {
                cudaMalloc(&this->data, size);
            }
            else
            {
                this->data = (Var *)malloc(size);
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
        }

        Tensor *Tensor::from_data(Shape shape, float *data)
        {
            Tensor *tensor = new Tensor(false, shape);
            for (int i = 0; i < tensor->count(); i++)
            {
                tensor->set_val(i, data[i]);
            }
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
            int dim_cnt = tensor->dims_count();

            if (dim_cnt == 1)
            {
                int cnt = tensor->shape[0];

                FILE *file_ptr = fopen(path, "w");

                fprintf(file_ptr, "col\n");

                for (int i = 0; i < cnt; i++)
                {
                    fprintf(file_ptr, "%f\n", tensor->get_var(i).v);
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
                            fprintf(file_ptr, "%f,", tensor->get_var(i * col_cnt + j).v);
                        }
                        else
                        {
                            fprintf(file_ptr, "%f", tensor->get_var(i * col_cnt + j).v);
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

        Tensor *Tensor::one_hot(Tensor *src)
        {
            int lst_dim_idx = src->dims_count() - 1;

            if (src->get_shape()[lst_dim_idx] != 1)
            {
                EPYON_CORE_THROW_ERROR("TENSOR ONE HOT ERROR: last dimension must be 1");
            }

            float min_val = src->min();

            if (min_val < 0.0f)
            {
                EPYON_CORE_THROW_ERROR("TENSOR ONE HOT ERROR: negative numbers not allowed");
            }

            int oh_dim = ((int)src->max()) + 1;

            std::vector<int> dst_dims = src->get_shape().get_dims();
            dst_dims[lst_dim_idx] = oh_dim;

            Tensor *dst = Tensor::zeros(src->is_cuda(), Shape(dst_dims));

            for (int i = 0; i < src->count(); i++)
            {
                int val = (int)src->get_var(i).v;
                dst->set_val(i * oh_dim + val, 1.0f);
            }

            return dst;
        }

        void Tensor::print_vec(Var *data, int cnt)
        {
            printf("[ ");
            for (int i = 0; i < cnt; i++)
            {
                float val = data[i].v;

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

        void Tensor::print_mtx(Var *data, int row_cnt, int col_cnt, const char *whitespace_str)
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

            switch (this->dims_count())
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
                    this->data = (Var *)malloc(src->size());
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
                        this->data = (Var *)malloc(src->size());
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
                    this->data = (Var *)malloc(this->size());
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
                Var *dst = (Var *)malloc(size);
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
                Var *dst;
                cudaMalloc(&dst, size);
                cudaMemcpy(dst, this->data, size, cudaMemcpyHostToDevice);
                free(this->data);
                this->data = dst;
                this->cuda = true;
            }
        }

        Shape Tensor::get_shape()
        {
            return this->shape;
        }

        Var *Tensor::get_data()
        {
            return this->data;
        }

        int Tensor::dims_count()
        {
            return this->shape.count();
        }

        int Tensor::count()
        {
            return this->shape.size();
        }

        size_t Tensor::size()
        {
            return sizeof(Var) * this->count();
        }

        float Tensor::min()
        {
            float min_val = FLT_MAX;

            float val = 0;

            for (int i = 0; i < this->count(); i++)
            {
                val = this->get_var(i).v;

                if (val < min_val)
                {
                    min_val = val;
                }
            }

            return min_val;
        }

        float Tensor::max()
        {
            float max_val = -FLT_MAX;

            float val = 0;

            for (int i = 0; i < this->count(); i++)
            {
                val = this->get_var(i).v;

                if (val > max_val)
                {
                    max_val = val;
                }
            }

            return max_val;
        }

        Var Tensor::get_var(int idx)
        {
            Var var;
            cudaMemcpy(&var, &this->data[idx], sizeof(Var), cudaMemcpyDefault);
            return var;
        }

        void Tensor::set_var(int idx, Var var)
        {
            cudaMemcpy(&this->data[idx], &var, sizeof(Var), cudaMemcpyDefault);
        }

        void Tensor::set_val(int idx, float val)
        {
            cudaMemcpy(&this->data[idx], &val, sizeof(float), cudaMemcpyDefault);
        }

        void Tensor::zeros()
        {
            this->fill(0.0f);
        }

        void Tensor::ones()
        {
            this->fill(1.0f);
        }

        void Tensor::fill(float val)
        {
            if (this->cuda)
            {
                k_set_all<<<(this->count() / EPYON_CORE_CUDA_THREADS_PER_BLOCK + 1), EPYON_CORE_CUDA_THREADS_PER_BLOCK>>>(this->data, this->count(), val);
            }
            else
            {
                for (int i = 0; i < this->count(); i++)
                {
                    this->set_val(i, val);
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
                this->set_val(i, d(gen));
            }

            if (orig_cuda)
            {
                this->to_cuda();
            }
        }

        AutoDiffContext::AutoDiffContext(bool cuda)
        {
            this->cuda = cuda;

            this->tape = (IntVar **)malloc(sizeof(IntVar *) * EPYON_AD_TAPE_BLOCK_CNT);

            this->block_cur = -1;
            this->elem_cur = -1;

            this->add_block();
        }

        AutoDiffContext::~AutoDiffContext()
        {
            if (this->cuda)
            {
                for (int i = 0; i < this->block_cur + 1; i++)
                {
                    cudaFree(this->tape[i]);
                }
            }
            else
            {
                for (int i = 0; i < this->block_cur + 1; i++)
                {
                    free(this->tape[i]);
                }
            }
            free(this->tape);
        }

        __host__ __device__ void AutoDiffContext::add_block()
        {
            this->block_cur++;
            if (this->cuda)
            {
                cudaMalloc(&this->tape[this->block_cur], sizeof(IntVar) * EPYON_AD_TAPE_BLOCK_SIZE);
            }
            else
            {
                this->tape[this->block_cur] = (IntVar *)malloc(sizeof(IntVar) * EPYON_AD_TAPE_BLOCK_SIZE);
            }

            this->elem_cur = -1;
        }

        __host__ __device__ TapeIndex AutoDiffContext::add_intermediate_variable(IntVar iv)
        {
            if (this->elem_cur != 0 && this->elem_cur >= EPYON_AD_TAPE_BLOCK_SIZE)
            {
                this->add_block();
            }
            this->elem_cur++;
            cudaMemcpy(&this->tape[this->block_cur][this->elem_cur], &iv, sizeof(IntVar), cudaMemcpyDefault);
            return {this->block_cur, this->elem_cur};
        }

        __host__ __device__ Var AutoDiffContext::op(float v, float pd1, float pd2, TapeIndex p1, TapeIndex p2)
        {
            Var var(v);
            var.i = this->add_intermediate_variable(IntVar(pd1, pd2, p1, p2));
            return var;
        }

        __host__ __device__ Var AutoDiffContext::var(float v)
        {
            Var var(v);
            var.i = this->add_intermediate_variable(IntVar());
            return v;
        }

        Tensor *AutoDiffContext::tensor(Tensor *tensor)
        {
            for (int i = 0; i < tensor->count(); i++)
            {
                tensor->set_var(i, Var(tensor->get_var(i).v, this->add_intermediate_variable(IntVar())));
            }

            return tensor;
        }

        void AutoDiffContext::backward()
        {
            if (this->cuda)
            {
            }
            else
            {
                this->tape[this->block_cur][this->elem_cur].d = 1.0f;

                int elem_cur = this->elem_cur;
                for (int i = this->block_cur; i >= 0; i--)
                {
                    for (int j = elem_cur; j >= 0; j--)
                    {
                        auto p1_i = this->tape[i][j].ps[0];
                        auto p2_i = this->tape[i][j].ps[1];

                        if (p1_i.block != EPYON_AD_INVALID_TAPE_BLOCK)
                            this->tape[p1_i.block][p1_i.elem].d += this->tape[p1_i.block][p1_i.elem].pds[0] * this->tape[i][j].d;
                        if (p2_i.block != EPYON_AD_INVALID_TAPE_BLOCK)
                            this->tape[p2_i.block][p2_i.elem].d += this->tape[p2_i.block][p2_i.elem].pds[0] * this->tape[i][j].d;
                    }
                    elem_cur = EPYON_AD_TAPE_BLOCK_SIZE;
                }
            }
        }

        __host__ __device__ Var AutoDiffContext::add(Var a, Var b)
        {
            return this->op(a.v + b.v,
                            1.0f, 1.0f,
                            a.i, b.i);
        }

        __host__ __device__ Var AutoDiffContext::mul(Var a, Var b)
        {
            return this->op(a.v * b.v,
                            b.v, a.v,
                            a.i, b.i);
        }

        __host__ __device__ Var AutoDiffContext::exp(Var a, float b)
        {
            return this->op(pow(a.v, b),
                            b * pow(a.v, b - 1), 0.0f,
                            a.i, EPYON_AD_DEFAULT_TAPE_INDEX);
        }

        void AutoDiffContext::sum(Tensor *src, Tensor *dst)
        {
            int cnt = src->count();

            if (this->cuda)
            {
                if (dst->dims_count() == 1)
                {
                    k_sum<<<cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK + 1, EPYON_CORE_CUDA_THREADS_PER_BLOCK>>>(this, src->get_data(), cnt, &dst->get_data()[0]);
                }
                else if (dst->dims_count() == 2)
                {
                    int src_offset = src->get_shape()[1];
                    for (int i = 0; i < dst->get_shape()[0]; i++)
                    {
                        k_sum<<<cnt / EPYON_CORE_CUDA_THREADS_PER_BLOCK + 1, EPYON_CORE_CUDA_THREADS_PER_BLOCK>>>(this, &src->get_data()[i * src_offset], cnt, &dst->get_data()[i]);
                    }
                }
            }
        }
    }
}