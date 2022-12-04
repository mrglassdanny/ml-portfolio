#include "eval.cuh"

__global__ void k_matmul_elemwise(float *in, float *w, float *out, int filter_cnt)
{
    int filter_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (filter_idx < filter_cnt && channel_idx < CHANNEL_CNT)
    {
        for (int i = 0; i < CHESS_ROW_CNT; i++)
        {
            for (int j = 0; j < CHESS_COL_CNT; j++)
            {
                atomicAdd(&out[(filter_idx * CHESS_BOARD_LEN) + (i * CHESS_COL_CNT) + j],
                          in[(channel_idx * CHESS_BOARD_LEN) + (i * CHESS_COL_CNT) + j] *
                              w[(filter_idx * CHANNEL_CNT * CHESS_BOARD_LEN) + (channel_idx * CHESS_BOARD_LEN) + (i * CHESS_COL_CNT) + j]);
            }
        }
    }
}

PosEvalModel::PosEvalModel(int w1_filter_cnt, int w2_filter_cnt)
{
    this->w1 = Tensor::random(true, Shape(w1_filter_cnt, CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT), 0.0f, 1.0f);
    this->w2 = Tensor::zeros(false, Shape(w2_filter_cnt, CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT));
}

PosEvalModel::~PosEvalModel()
{
    delete this->w1;
    delete this->w2;
}

Tensor *PosEvalModel::forward(Tensor *x)
{
    Tensor *p = Tensor::zeros(true, Shape(this->w1->shape()[0], CHESS_ROW_CNT, CHESS_COL_CNT));

    int grid_row_cnt = (this->w1->shape()[0] / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;
    int grid_col_cnt = (CHANNEL_CNT / ZERO_CORE_CUDA_THREADS_PER_BLOCK) + 1;

    dim3 grid_dims(grid_col_cnt, grid_row_cnt);
    dim3 block_dims(ZERO_CORE_CUDA_THREADS_PER_BLOCK, ZERO_CORE_CUDA_THREADS_PER_BLOCK);

    k_matmul_elemwise<<<grid_dims, block_dims>>>(x->data(), this->w1->data(), p->data(), this->w1->shape()[0]);

    x->print();
    this->w1->print();
    p->print();

    return p;
}