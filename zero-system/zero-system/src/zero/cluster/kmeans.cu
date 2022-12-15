#include "kmeans.cuh"

#define THREADS_PER_BLOCK 32
#define MAX_CLUSTER_CNT 8192

using namespace zero::core;
using namespace zero::cluster;

float __device__ d_get_loss(float x_val, float cluster_val)
{
    return ((x_val - cluster_val) * (x_val - cluster_val));
}

int __device__ d_get_min(float *arr, int cnt)
{
    int min_idx = 0;
    float min_val = FLT_MAX;

    for (int i = 0; i < cnt; i++)
    {
        float cur_val = arr[i];
        if (cur_val < min_val)
        {
            min_idx = i;
            min_val = cur_val;
        }
    }

    return min_idx;
}

void __global__ k_assign_to_clusters(float *x_arr, float *cluster_assignments_arr, float *cluster_arr, float *loss_val, int feature_cnt, int cluster_cnt, int batch_size)
{
    float temp[MAX_CLUSTER_CNT];
    memset(temp, 0, sizeof(float) * MAX_CLUSTER_CNT);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batch_size)
    {
        for (int feature_idx = 0; feature_idx < feature_cnt; feature_idx++)
        {
            for (int cluster_idx = 0; cluster_idx < cluster_cnt; cluster_idx++)
            {
                temp[cluster_idx] += d_get_loss(x_arr[tid * feature_cnt + feature_idx], cluster_arr[cluster_idx * feature_cnt + feature_idx]);
            }
        }

        for (int cluster_idx = 0; cluster_idx < cluster_cnt; cluster_idx++)
        {
            temp[cluster_idx] = sqrt(temp[cluster_idx]);
        }

        int min_cluster_idx = d_get_min(temp, cluster_cnt);

        cluster_assignments_arr[tid] = min_cluster_idx;

        if (loss_val != nullptr)
        {
            atomicAdd(loss_val, temp[min_cluster_idx]);
        }
    }
}

void __global__ k_update_clusters_part_1(float *x_arr, float *cluster_assignments_arr, float *cluster_arr, float *cluster_assignment_cnts_arr, int feature_cnt, int cluster_cnt, int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batch_size)
    {
        int cluster_idx = cluster_assignments_arr[tid];

        atomicAdd(&cluster_assignment_cnts_arr[cluster_idx], 1.0f);

        for (int feature_idx = 0; feature_idx < feature_cnt; feature_idx++)
        {
            atomicAdd(&cluster_arr[cluster_idx * feature_cnt + feature_idx], x_arr[tid * feature_cnt + feature_idx]);
        }
    }
}

void __global__ k_update_clusters_part_2(float *cluster_arr, float *cluster_assignment_cnts_arr, int cluster_cnt, int feature_cnt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < cluster_cnt * feature_cnt)
    {
        int cluster_idx = tid / feature_cnt;

        cluster_arr[tid] /= (cluster_assignment_cnts_arr[cluster_idx]);
    }
}

KMeans::KMeans()
{
    this->cluster_cnt_ = 0;
    this->feature_cnt_ = 0;
    this->clusters_ = nullptr;
}

KMeans::KMeans(int cluster_cnt, int feature_cnt)
{
    this->cluster_cnt_ = cluster_cnt;
    this->feature_cnt_ = feature_cnt;

    this->clusters_ = Tensor::zeros(false, Shape(cluster_cnt, feature_cnt));
}

KMeans::KMeans(const KMeans &src)
{
    this->cluster_cnt_ = src.cluster_cnt_;
    this->feature_cnt_ = src.feature_cnt_;
    this->clusters_ = new Tensor(*src.clusters_);
}

KMeans::~KMeans()
{
    delete this->clusters_;
}

void KMeans::load(const char *path)
{
    FILE *file_ptr = fopen(path, "rb");

    fread(&this->cluster_cnt_, sizeof(int), 1, file_ptr);
    fread(&this->feature_cnt_, sizeof(int), 1, file_ptr);

    int tot_cnt = (this->cluster_cnt_ * this->feature_cnt_);
    float *cluster_buf = (float *)malloc(sizeof(float) * tot_cnt);
    fread(cluster_buf, sizeof(float), tot_cnt, file_ptr);
    this->clusters_ = Tensor::from_data(Shape(this->cluster_cnt_, this->feature_cnt_), cluster_buf);
    this->clusters_->to_cuda();

    free(cluster_buf);

    fclose(file_ptr);
}

void KMeans::save(const char *path)
{
    FILE *file_ptr = fopen(path, "wb");

    this->clusters_->to_cpu();

    fwrite(&this->cluster_cnt_, sizeof(int), 1, file_ptr);
    fwrite(&this->feature_cnt_, sizeof(int), 1, file_ptr);
    fwrite(this->clusters_->data(), sizeof(float), (this->cluster_cnt_ * this->feature_cnt_), file_ptr);

    fclose(file_ptr);
}

void KMeans::initialize(Tensor *x)
{
    this->validate_input(x);

    this->clusters_->to_cpu();

    std::vector<int> rand_nums;
    rand_nums.reserve(this->cluster_cnt_);

    int batch_size = x->shape()[0];

    for (int i = 0; i < this->cluster_cnt_; i++)
    {
        bool rand_num_already_added;
        int rand_num;

        do
        {
            rand_num_already_added = false;
            rand_num = rand() % batch_size;

            for (int j = 0; j < rand_nums.size(); j++)
            {
                if (rand_nums[j] == rand_num)
                {
                    rand_num_already_added = true;
                    break;
                }
            }

        } while (rand_num_already_added);

        rand_nums.push_back(rand_num);
    }

    for (int cluster_idx = 0; cluster_idx < this->cluster_cnt_; cluster_idx++)
    {
        int rand_row_idx = rand_nums[cluster_idx];
        cudaMemcpy(&this->clusters_->data()[cluster_idx * this->feature_cnt_],
                   &x->data()[rand_row_idx * this->feature_cnt_], sizeof(float) * this->feature_cnt_, cudaMemcpyDefault);
    }

    this->clusters_->to_cuda();
    x->to_cuda();
}

float KMeans::train(Tensor *x)
{
    this->initialize(x);

    int batch_size = x->shape()[0];

    Tensor *cluster_assignments = Tensor::zeros(true, Shape(batch_size, 1));
    Tensor *cluster_assignment_cnts = Tensor::zeros(true, Shape(this->cluster_cnt_, 1));

    int epoch = 1;

    float h_loss_val;
    float h_prv_loss_val = FLT_MAX;

    float *d_loss_val;
    cudaMalloc(&d_loss_val, sizeof(float));
    cudaMemset(d_loss_val, 0, sizeof(float));

    while (true)
    {
        // Assign inputs to clusters:
        {
            int num_blocks((batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1);
            k_assign_to_clusters<<<num_blocks, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(x->data(), cluster_assignments->data(), this->clusters_->data(), d_loss_val, this->feature_cnt_, this->cluster_cnt_, batch_size);
        }

        // Analyze loss:
        {
            cudaMemcpy(&h_loss_val, d_loss_val, sizeof(float), cudaMemcpyDeviceToHost);

            h_loss_val /= batch_size;

            if (h_prv_loss_val <= h_loss_val)
            {
                break;
            }

            h_prv_loss_val = h_loss_val;
        }

        // Reset clusters prior to update:
        this->clusters_->zeros();

        // Update clusters part 1:
        {
            int num_blocks((batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1);
            k_update_clusters_part_1<<<num_blocks, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(x->data(), cluster_assignments->data(), this->clusters_->data(),
                                                                                       cluster_assignment_cnts->data(), this->feature_cnt_, this->cluster_cnt_, batch_size);
        }

        // Update clusters part 2:
        {
            int num_blocks(((this->cluster_cnt_ * this->feature_cnt_) / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1);
            k_update_clusters_part_2<<<num_blocks, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(this->clusters_->data(), cluster_assignment_cnts->data(), this->cluster_cnt_, this->feature_cnt_);
        }

        // Reset loss and assignment counts for next epoch:
        {
            cudaMemset(d_loss_val, 0, sizeof(float));

            cluster_assignment_cnts->zeros();
        }

        epoch++;
    }

    cudaFree(d_loss_val);

    delete cluster_assignments;
    delete cluster_assignment_cnts;

    return h_loss_val;
}

Tensor *KMeans::predict(Tensor *x)
{
    int batch_size = x->shape()[0];

    Tensor *cluster_assignments = new Tensor(true, Shape(batch_size, 1));

    {
        int num_blocks((batch_size / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1);
        k_assign_to_clusters<<<num_blocks, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(x->data(), cluster_assignments->data(), this->clusters_->data(), nullptr,
                                                                               this->feature_cnt_, this->cluster_cnt_, batch_size);
    }

    cluster_assignments->to_cpu();

    return cluster_assignments;
}

void KMeans::validate_input(Tensor *x)
{
    if (this->feature_cnt_ != (x->dims_size() / x->shape()[0]))
    {
        ZERO_CORE_THROW_ERROR("KMEANS VALIDATION FAILED: invalid input shape");
    }
}

Tensor *KMeans::clusters()
{
    return this->clusters_;
}

float KMeans::save_best(Tensor *x, int cluster_cnt, int iter_cnt, const char *path)
{
    int batch_size = x->shape()[0];
    int feature_cnt = x->dims_size() / batch_size;

    KMeans *kmeans = new KMeans(cluster_cnt, feature_cnt);
    KMeans *best_kmeans = new KMeans(cluster_cnt, feature_cnt);

    float loss;
    float min_loss = FLT_MAX;

    for (int i = 0; i < iter_cnt; i++)
    {
        loss = kmeans->train(x);

        if (loss < min_loss)
        {
            best_kmeans->clusters_->copy(kmeans->clusters_);

            min_loss = loss;
        }

        kmeans->clusters_->zeros();
    }

    best_kmeans->save(path);

    delete kmeans;
    delete best_kmeans;

    return min_loss;
}

void KMeans::elbow_analysis(Tensor *x, int cluster_cnt_lower, int cluster_cnt_upper,
                            int iter_cnt, const char *csv_path)
{
    FILE *csv_file_ptr = fopen(csv_path, "w");
    fprintf(csv_file_ptr, "cluster_cnt,min_loss\n");

    int batch_size = x->shape()[0];
    int feature_cnt = x->dims_size() / batch_size;

    for (int cluster_cnt = cluster_cnt_lower; cluster_cnt < cluster_cnt_upper; cluster_cnt++)
    {
        KMeans *kmeans = new KMeans(cluster_cnt, feature_cnt);
        KMeans *best_kmeans = new KMeans(cluster_cnt, feature_cnt);

        float loss;
        float min_loss = FLT_MAX;

        for (int i = 0; i < iter_cnt; i++)
        {
            loss = kmeans->train(x);

            if (loss < min_loss)
            {
                best_kmeans->clusters_->copy(kmeans->clusters_);

                min_loss = loss;
            }

            kmeans->clusters_->zeros();
        }

        printf("CLUSTERS: %d\tLOSS: %f\n", cluster_cnt, min_loss);

        fprintf(csv_file_ptr, "%d,%f\n", cluster_cnt, min_loss);

        delete kmeans;
        delete best_kmeans;
    }

    fclose(csv_file_ptr);
}