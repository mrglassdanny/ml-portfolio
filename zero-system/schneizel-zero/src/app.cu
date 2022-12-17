#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

#include "chess.cuh"

using namespace zero::core;
using namespace zero::nn;
using namespace zero::cluster;
using namespace chess;

__global__ void k_co_weight_step(float *w, float *dw, float *mdw, int w_cnt, float lr, float beta1, int step_num, int batch_size)
{
    int w_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (w_elem_idx < w_cnt)
    {
        mdw[w_elem_idx] = beta1 * mdw[w_elem_idx] + (1.0f - beta1) * dw[w_elem_idx];

        float corrected_mdw = mdw[w_elem_idx] / (1.0f - pow(beta1, step_num));

        w[w_elem_idx] -= (lr * corrected_mdw / batch_size);
        if (w[w_elem_idx] < 0.0f)
        {
            w[w_elem_idx] = 0.0f;
        }
    }
}

__global__ void k_co_bias_step(float *b, float *db, float *mdb, int b_cnt, float lr, float beta1, int step_num, int batch_size)
{
    int b_elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_elem_idx < b_cnt)
    {
        mdb[b_elem_idx] = beta1 * mdb[b_elem_idx] + (1.0f - beta1) * db[b_elem_idx];

        float corrected_mdb = mdb[b_elem_idx] / (1.0f - pow(beta1, step_num));

        b[b_elem_idx] -= (lr * corrected_mdb / batch_size);
        if (b[b_elem_idx] < 0.0f)
        {
            b[b_elem_idx] = 0.0f;
        }
    }
}

class ChessOptimizer : public Optimizer
{
private:
    float beta1_;
    std::vector<Tensor *> mdws_;
    std::vector<Tensor *> mdbs_;

public:
    ChessOptimizer(std::vector<Parameters *> model_params, float learning_rate, float beta1)
        : Optimizer(model_params, learning_rate)
    {
        this->beta1_ = beta1;

        for (Parameters *params : model_params)
        {
            this->mdws_.push_back(Tensor::zeros(true, params->weight_gradients()->shape()));
            this->mdbs_.push_back(Tensor::zeros(true, params->bias_gradients()->shape()));
        }
    }

    ~ChessOptimizer()
    {
        for (int i = 0; i < this->mdws_.size(); i++)
        {
            delete this->mdws_[i];
            delete this->mdbs_[i];
        }
    }

    void step(int batch_size)
    {
        for (int i = 0; i < this->model_params_.size(); i++)
        {
            Parameters *params = this->model_params_[i];

            Tensor *w = params->weights();
            Tensor *b = params->biases();
            Tensor *dw = params->weight_gradients();
            Tensor *db = params->bias_gradients();
            Tensor *mdw = this->mdws_[i];
            Tensor *mdb = this->mdbs_[i];

            int w_cnt = w->count();
            int b_cnt = b->count();

            k_co_weight_step<<<w_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(w->data(), dw->data(), mdw->data(),
                                                                                                                 w_cnt, this->lr_, this->beta1_, this->step_num_, batch_size);
            k_co_bias_step<<<b_cnt / ZERO_CORE_CUDA_THREADS_PER_BLOCK + 1, ZERO_CORE_CUDA_THREADS_PER_BLOCK>>>(b->data(), db->data(), mdb->data(),
                                                                                                               b_cnt, this->lr_, this->beta1_, this->step_num_, batch_size);
        }

        this->step_num_++;
    }

    Optimizer *ChessOptimizer::copy()
    {
        return new ChessOptimizer(this->model_params_, this->lr_, this->beta1_);
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

    int white_win_cnt = 0;
    int black_win_cnt = 0;

    for (auto pgn_game : pgn_games)
    {
        // Only save games where there was a winner.
        if (pgn_game->lbl != 0)
        {
            // Try to make sure that we have the same amount of white wins to black wins.
            if ((white_win_cnt <= 5000 && pgn_game->lbl == 1) || (black_win_cnt <= 5000 && pgn_game->lbl == -1))
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

                if (pgn_game->lbl == 1)
                {
                    white_win_cnt++;
                }
                else
                {
                    black_win_cnt++;
                }
            }
        }

        delete pgn_game;
    }

    fclose(train_data_file);
    fclose(train_lbl_file);
    fclose(test_data_file);
    fclose(test_lbl_file);

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

void train_n_test(Model *model, int epochs, std::vector<Batch> *train_ds, std::vector<Batch> *test_ds)
{
    int train_batch_cnt = train_ds->size();
    int test_batch_cnt = test_ds->size();

    // Train:
    {
        FILE *train_csv = fopen("temp/train.csv", "w");
        fprintf(train_csv, "epoch,batch,loss,accuracy\n");

        bool quit = false;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch_idx = 0; batch_idx < train_batch_cnt; batch_idx++)
            {
                auto batch = &train_ds->at(batch_idx);
                auto x = batch->x;
                auto y = batch->y;

                auto p = model->forward(x);

                float loss = model->loss(p, y);
                float acc = model->accuracy(p, y, Model::regression_tanh_accuracy_fn);
                fprintf(train_csv, "%d,%d,%f,%f\n", epoch, batch_idx, loss, acc);

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
        }

        fclose(train_csv);
    }

    // Test:
    {
        float loss = 0.0f;
        float acc = 0.0f;

        for (int batch_idx = 0; batch_idx < test_batch_cnt; batch_idx++)
        {
            auto batch = &test_ds->at(batch_idx);
            auto x = batch->x;
            auto y = batch->y;

            auto p = model->forward(x);

            loss += model->loss(p, y);
            acc += model->accuracy(p, y, Model::regression_tanh_accuracy_fn);

            if (batch_idx == test_batch_cnt - 1)
            {
                p->print();
                y->print();
            }

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
        model->set_initializer(new Xavier());
        model->hadamard_product(x_shape, 4, new Tanh());
        model->hadamard_product(4, new Tanh());
        model->matrix_product(4, new Tanh());
        model->matrix_product(4, new Tanh());
        model->linear(y_shape, new Tanh());
        model->set_loss(new MSE());

        model->summarize();
        model->validate_gradients(x, y, false);

        delete model;
    }

    {
        auto model = new Model();
        model->set_initializer(new ChessInitializer());
        model->hadamard_product(x_shape, 4, new Tanh());
        model->matrix_product(4, new Tanh());
        model->linear(128, new Tanh());
        model->linear(32, new Tanh());
        model->linear(y_shape, new Tanh());
        model->set_loss(new MSE());

        model->summarize();
        model->validate_gradients(x, y, false);

        delete model;
    }

    {
        auto model = new Model();
        model->set_initializer(new Xavier());
        model->hadamard_product(x_shape, 4, new Tanh());
        model->hadamard_product(4, new Tanh());
        model->matrix_product(4, new Tanh());
        model->matrix_product(4, new Tanh());
        model->linear(y_shape, new Tanh());
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
    auto train_ds = get_train_dataset(256);
    auto test_ds = get_test_dataset(256);

    Shape x_shape = train_ds[0].x->shape();
    Shape y_shape = train_ds[0].y->shape();

    {
        auto model = new Model(new ChessInitializer());

        model->hadamard_product(x_shape, 32, new Tanh());
        model->matrix_product(32, new Tanh());
        model->linear(256, new Tanh());
        model->linear(64, new Tanh());
        model->linear(y_shape, new Tanh());

        model->set_loss(new MSE());
        model->set_optimizer(new ChessOptimizer(model->parameters(), 0.01f, ZERO_NN_BETA_1));

        model->summarize();

        train_n_test(model, epochs, &train_ds, &test_ds);

        delete model;
    }

    {
        auto model = new Model(new ChessInitializer());

        model->hadamard_product(x_shape, 32, new Tanh());
        model->linear(256, new Tanh());
        model->linear(64, new Tanh());
        model->linear(y_shape, new Tanh());

        model->set_loss(new MSE());
        model->set_optimizer(new ChessOptimizer(model->parameters(), 0.01f, ZERO_NN_BETA_1));

        model->summarize();

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

int main()
{
    srand(time(NULL));

    // export_pgn("data/data.pgn");

    // grad_tests();

    compare_models(5);

    // self_play(3, 3, true);

    return 0;
}