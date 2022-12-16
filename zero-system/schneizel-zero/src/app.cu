#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

#include "chess.cuh"

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;
using namespace chess;

__global__ void k_co_weight_step(float *w, float *dw, int w_cnt, float lr, int batch_size)
{
    int w_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (w_elem_idx < w_cnt)
    {
        w[w_elem_idx] -= (lr * dw[w_elem_idx] / batch_size);
        if (w[w_elem_idx] < 0.0f)
        {
            w[w_elem_idx] = 0.0f;
        }
    }
}

__global__ void k_co_bias_step(float *b, float *db, int b_cnt, float lr, int batch_size)
{
    int b_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_elem_idx < b_cnt)
    {
        b[b_elem_idx] -= (lr * db[b_elem_idx] / batch_size);
        if (b[b_elem_idx] < 0.0f)
        {
            b[b_elem_idx] = 0.0f;
        }
    }
}

class ChessOptimizer : public Optimizer
{
public:
    ChessOptimizer(std::vector<Parameters *> model_params, float learning_rate)
        : Optimizer(model_params, learning_rate)
    {
    }

    void ChessOptimizer::step(int batch_size)
    {
        for (Parameters *params : this->model_params_)
        {
            Tensor *w = params->weights();
            Tensor *b = params->biases();
            Tensor *dw = params->weight_gradients();
            Tensor *db = params->bias_gradients();

            int w_cnt = w->count();
            int b_cnt = b->count();

            k_co_weight_step<<<w_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(w->data(), dw->data(), w_cnt, this->lr_, batch_size);
            k_co_bias_step<<<b_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), b_cnt, this->lr_, batch_size);
        }

        this->step_num_++;
    }

    Optimizer *ChessOptimizer::copy()
    {
        return new SGD(this->model_params_, this->lr_);
    }
};

class ChessInitializer : public Initializer
{
public:
    void initialize(Tensor *tensor, int fan_in, int fan_out)
    {
        tensor->random(0.0f, sqrt(0.5f / fan_in));
        tensor->abs();
    }

    Initializer *copy()
    {
        return new ChessInitializer();
    }
};

struct ClusterData
{
    KMeans *model;
    Tensor *white_win_cnts;
    Tensor *black_win_cnts;
};

struct Game
{
    std::vector<Board> boards;
    float lbl;
};

Game self_play(int white_depth, int black_depth, bool print)
{
    Board board;
    Move prev_move;

    Game game;
    game.lbl = 0.0f;

    int move_cnt = 0;

    while (move_cnt < 200)
    {
        if (print)
        {
            printf("\nWHITE TURN\n");
            if (move_cnt == 0)
            {
                board.print();
            }
            else
            {
                board.print(prev_move);
            }
        }

        if (board.is_checkmate(false, true))
        {
            if (print)
                printf("WHITE CHECKMATED!\n");

            game.lbl = -1.0f;
            break;
        }
        else if (!board.has_moves(true))
        {
            if (print)
                printf("WHITE STALEMATED!\n");

            break;
        }

        auto evals = board.minimax_alphabeta_dyn(true, white_depth);
        printf("Ties: %d\n", evals.size());
        int r = rand() % evals.size();
        board.change(evals[r].move);
        prev_move = evals[r].move;
        Board cpy_board;
        cpy_board.copy(&board);
        game.boards.push_back(cpy_board);

        move_cnt++;

        if (print)
        {
            printf("\nBLACK TURN\n");
            board.print(prev_move);
        }

        if (board.is_checkmate(true, true))
        {
            if (print)
                printf("BLACK CHECKMATED!\n");

            game.lbl = 1.0f;
            break;
        }
        else if (!board.has_moves(false))
        {
            if (print)
                printf("BLACK STALEMATED!\n");

            break;
        }

        evals = board.minimax_alphabeta(false, black_depth);
        printf("Ties: %d\n", evals.size());
        r = rand() % evals.size();
        board.change(evals[r].move);
        prev_move = evals[r].move;
        Board cpy_board2;
        cpy_board2.copy(&board);
        game.boards.push_back(cpy_board2);

        move_cnt++;
    }

    return game;
}

// NOTE: excludes ties and openings.
void export_pgn(const char *path)
{
    auto pgn_games = PGN::import(path);

    int game_cnt = 0;
    long move_cnt = 0;

    FILE *train_data_file = fopen("temp/train.data", "wb");
    FILE *train_lbl_file = fopen("temp/train.lbl", "wb");
    FILE *test_data_file = fopen("temp/test.data", "wb");
    FILE *test_lbl_file = fopen("temp/test.lbl", "wb");

    float data_buf[CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT];
    float lbl_buf;

    for (auto pgn_game : pgn_games)
    {
        // Only save games where there was a winner.
        if (pgn_game->lbl != 0)
        {
            Board board;
            bool white = true;

            int game_move_cnt = 0;

            for (auto move_str : pgn_game->move_strs)
            {
                auto move = board.change(move_str, white);
                white = !white;

                // Skip openings.
                if (game_move_cnt > 6)
                {
                    board.one_hot_encode(data_buf);
                    lbl_buf = (float)pgn_game->lbl;

                    if (rand() % 20 == 0)
                    {
                        fwrite(data_buf, sizeof(data_buf), 1, test_data_file);
                        fwrite(&lbl_buf, sizeof(lbl_buf), 1, test_lbl_file);
                    }
                    else
                    {
                        fwrite(data_buf, sizeof(data_buf), 1, train_data_file);
                        fwrite(&lbl_buf, sizeof(lbl_buf), 1, train_lbl_file);
                    }

                    move_cnt++;
                }

                game_move_cnt++;
            }
            game_cnt++;
        }

        delete pgn_game;
    }

    fclose(train_data_file);
    fclose(train_lbl_file);
    fclose(test_data_file);
    fclose(test_lbl_file);

    printf("GAME COUNT: %d\tMOVE COUNT: %ld\n", game_cnt, move_cnt);
}

// NOTE: excludes ties.
void export_pgn_cluster(const char *path)
{
    auto pgn_games = PGN::import(path);

    int game_cnt = 0;
    long move_cnt = 0;

    FILE *data_file = fopen("temp/cluster.data", "wb");
    FILE *lbl_file = fopen("temp/cluster.lbl", "wb");

    float data_buf[CHESS_ROW_CNT * CHESS_COL_CNT];
    float lbl_buf;

    for (auto pgn_game : pgn_games)
    {
        // if (game_cnt < 1000)
        {
            // Only save games where there was a winner.
            if (pgn_game->lbl != 0)
            {
                Board board;
                bool white = true;

                for (auto move_str : pgn_game->move_strs)
                {
                    auto move = board.change(move_str, white);
                    white = !white;

                    board.material_encode(data_buf);
                    lbl_buf = (float)pgn_game->lbl;

                    fwrite(data_buf, sizeof(data_buf), 1, data_file);
                    fwrite(&lbl_buf, sizeof(lbl_buf), 1, lbl_file);

                    move_cnt++;
                }

                game_cnt++;
            }
        }

        delete pgn_game;
    }

    fclose(data_file);
    fclose(lbl_file);

    printf("GAME COUNT: %d\tMOVE COUNT: %ld\n", game_cnt, move_cnt);
}

struct Batch
{
    zero::core::Tensor *x;
    zero::core::Tensor *y;
};

std::vector<Batch> get_train_dataset(int batch_size)
{
    int oh_board_len = CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT;
    int oh_board_size = oh_board_len * sizeof(float);

    long long data_file_size = zero::core::FileUtils::get_file_size("temp/train.data");
    size_t data_cnt = data_file_size / oh_board_size;

    std::vector<Batch> batches;

    FILE *data_file = fopen("temp/train.data", "rb");
    FILE *lbl_file = fopen("temp/train.lbl", "rb");

    float *data_buf = (float *)malloc(data_file_size);
    fread(data_buf, 1, (data_file_size), data_file);

    float *lbl_buf = (float *)malloc(sizeof(float) * data_cnt);
    fread(lbl_buf, 1, (sizeof(float) * data_cnt), lbl_file);

    fclose(data_file);
    fclose(lbl_file);

    for (int i = 0; i < data_cnt / batch_size; i++)
    {
        auto x = zero::core::Tensor::from_data(zero::core::Shape(batch_size, CHESS_BOARD_CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT), &data_buf[i * batch_size * oh_board_len]);
        auto y = zero::core::Tensor::from_data(zero::core::Shape(batch_size, 1), &lbl_buf[i * batch_size]);

        batches.push_back({x, y});
    }

    free(data_buf);
    free(lbl_buf);

    return batches;
}

std::vector<Batch> get_test_dataset(int batch_size)
{
    int oh_board_len = CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT;
    int oh_board_size = oh_board_len * sizeof(float);

    long long data_file_size = zero::core::FileUtils::get_file_size("temp/test.data");
    size_t data_cnt = data_file_size / oh_board_size;

    std::vector<Batch> batches;

    FILE *data_file = fopen("temp/test.data", "rb");
    FILE *lbl_file = fopen("temp/test.lbl", "rb");

    float *data_buf = (float *)malloc(data_file_size);
    fread(data_buf, oh_board_size, data_cnt, data_file);

    float *lbl_buf = (float *)malloc(sizeof(float) * data_cnt);
    fread(lbl_buf, sizeof(float), data_cnt, lbl_file);

    fclose(data_file);
    fclose(lbl_file);

    for (int i = 0; i < data_cnt / batch_size; i++)
    {
        auto x = zero::core::Tensor::from_data(zero::core::Shape(batch_size, CHESS_BOARD_CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT), &data_buf[i * batch_size * oh_board_len]);
        auto y = zero::core::Tensor::from_data(zero::core::Shape(batch_size, 1), &lbl_buf[i * batch_size]);

        batches.push_back({x, y});
    }

    free(data_buf);
    free(lbl_buf);

    return batches;
}

Batch get_cluster_dataset()
{
    int mat_board_len = CHESS_ROW_CNT * CHESS_COL_CNT;
    int mat_board_size = mat_board_len * sizeof(float);

    long long data_file_size = zero::core::FileUtils::get_file_size("temp/cluster.data");
    size_t data_cnt = data_file_size / mat_board_size;
    std::vector<Batch> batches;

    FILE *data_file = fopen("temp/cluster.data", "rb");
    FILE *lbl_file = fopen("temp/cluster.lbl", "rb");

    float *data_buf = (float *)malloc(data_file_size);
    fread(data_buf, mat_board_size, data_cnt, data_file);

    float *lbl_buf = (float *)malloc(sizeof(float) * data_cnt);
    fread(lbl_buf, sizeof(float), data_cnt, lbl_file);

    fclose(data_file);
    fclose(lbl_file);

    auto x = Tensor::from_data(Shape(data_cnt, CHESS_ROW_CNT, CHESS_COL_CNT), data_buf);
    auto y = Tensor::from_data(Shape(data_cnt, 1), lbl_buf);

    free(data_buf);
    free(lbl_buf);

    return Batch{x, y};
}

void train_n_test(Model *model, int epochs, std::vector<Batch> *train_ds, std::vector<Batch> *test_ds)
{
    int train_batch_cnt = train_ds->size();
    int test_batch_cnt = test_ds->size();

    bool quit = false;

    // Train:
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int j = 0; j < train_batch_cnt; j++)
        {
            auto batch = &train_ds->at(j);
            auto x = batch->x;
            auto y = batch->y;

            auto p = model->forward(x);

            model->backward(p, y);
            model->step();

            delete p;

            if (_kbhit())
            {
                if (_getch() == 'q')
                {
                    quit = true;
                    break;
                }
            }
        }

        if (quit)
        {
            break;
        }

        printf("EPOCH COMPLETE: %d\n", epoch);
    }

    // Test:
    {
        float loss = 0.0f;
        float acc = 0.0f;

        for (int j = 0; j < test_batch_cnt; j++)
        {
            auto batch = &test_ds->at(j);
            auto x = batch->x;
            auto y = batch->y;

            auto p = model->forward(x);
            loss += model->loss(p, y);
            acc += model->accuracy(p, y, Model::regression_tanh_accuracy_fn);
            delete p;
        }

        float test_acc_pct = (acc / (float)test_batch_cnt) * 100.0f;

        model->summarize();
        printf("TEST LOSS: %f\tTEST ACCURACY: %f%%\n",
               (loss / (float)test_batch_cnt),
               test_acc_pct);
    }
}

void grad_tests()
{
    auto test_ds = get_test_dataset(1);

    auto x = test_ds[0].x;
    auto y = test_ds[0].y;

    Shape x_shape = x->shape();
    Shape y_shape = y->shape();

    {
        auto model = new Model();
        model->set_initializer(new XavierInitializer());
        model->hadamard_product(x_shape, 4, new TanhActivation());
        model->hadamard_product(4, new TanhActivation());
        model->matrix_product(4, new TanhActivation());
        model->matrix_product(4, new TanhActivation());
        model->linear(y_shape, new TanhActivation());
        model->set_loss(new MSE());

        model->summarize();
        model->validate_gradients(x, y, true);

        delete model;
    }

    {
        auto model = new Model();
        model->set_initializer(new ChessInitializer());
        model->hadamard_product(x_shape, 4, new TanhActivation());
        model->hadamard_product(4, new TanhActivation());
        model->matrix_product(4, new TanhActivation());
        model->matrix_product(4, new TanhActivation());
        model->linear(y_shape, new TanhActivation());
        model->set_loss(new MSE());

        model->summarize();
        model->validate_gradients(x, y, false);

        delete model;
    }

    {
        auto model = new Model();
        model->set_initializer(new XavierInitializer());
        model->hadamard_product(x_shape, 4, new TanhActivation());
        model->hadamard_product(4, new TanhActivation());
        model->matrix_product(4, new TanhActivation());
        model->matrix_product(4, new TanhActivation());
        model->linear(y_shape, new TanhActivation());
        model->set_loss(new MSE());

        model->summarize();
        model->validate_gradients(x, y, false);

        auto cpy = model->copy();
        cpy->validate_gradients(x, y, false);

        delete model;
        delete cpy;
    }
}

void compare_models(int epochs)
{
    auto train_ds = get_train_dataset(64);
    auto test_ds = get_test_dataset(64);

    Shape x_shape = train_ds[0].x->shape();
    Shape y_shape = train_ds[0].y->shape();

    {
        printf("\n\n");
        auto model = new Model(new XavierInitializer());
        model->hadamard_product(x_shape, 16, new TanhActivation());
        model->matrix_product(16, new TanhActivation());
        model->linear(y_shape, new TanhActivation());
        model->set_loss(new MSE());
        model->set_optimizer(new SGD(model->parameters(), 0.1f));

        train_n_test(model, epochs, &train_ds, &test_ds);

        delete model;
    }

    {
        printf("\n\n");
        auto model = new Model(new ChessInitializer());
        model->hadamard_product(x_shape, 16, new TanhActivation());
        model->matrix_product(16, new TanhActivation());
        model->linear(y_shape, new TanhActivation());
        model->set_loss(new MSE());
        model->set_optimizer(new ChessOptimizer(model->parameters(), 0.1f));

        train_n_test(model, epochs, &train_ds, &test_ds);

        delete model;
    }

    for (auto batch : train_ds)
    {
        delete batch.x;
        delete batch.y;
    }

    for (auto batch : test_ds)
    {
        delete batch.x;
        delete batch.y;
    }
}

ClusterData cluster_tests()
{
    auto batch = get_cluster_dataset();

    auto model = KMeans::save_best(batch.x, 100000, 3, "temp/model.km");

    auto preds = model->predict(batch.x);

    auto white_win_cnts = Tensor::zeros(false, Shape(model->clusters()->shape()[0], 1));
    auto black_win_cnts = Tensor::zeros(false, Shape(model->clusters()->shape()[0], 1));

    for (int i = 0; i < preds->count(); i++)
    {
        int cluster_idx = (int)preds->get_val(i);
        if (batch.y->get_val(i) == 1.0f)
        {
            white_win_cnts->set_val(cluster_idx, white_win_cnts->get_val(cluster_idx) + 1);
        }
        else if (batch.y->get_val(i) == -1.0f)
        {
            black_win_cnts->set_val(cluster_idx, black_win_cnts->get_val(cluster_idx) + 1);
        }
    }

    delete preds;

    delete batch.x;
    delete batch.y;

    return ClusterData{model, white_win_cnts, black_win_cnts};
}

int main()
{
    srand(time(NULL));

    // export_pgn("data/data.pgn");
    // export_pgn_cluster("data/data.pgn");

    // grad_tests();

    // compare_models(4);

    // auto data = cluster_tests();

    // data.white_win_cnts->print();
    // data.black_win_cnts->print();

    self_play(3, 3, true);

    return 0;
}