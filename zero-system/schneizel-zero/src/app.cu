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
        tensor->random(0.0f, sqrt(1.0f / fan_in));
        tensor->abs();
    }

    Initializer *copy()
    {
        return new ChessInitializer();
    }
};

int chess_tanh_accuracy_fn(Tensor *p, Tensor *y, int batch_size)
{
    int correct_cnt = 0;

    for (int i = 0; i < batch_size; i++)
    {
        float p_val = p->get_val(i);
        if (p_val < -0.25f)
        {
            p_val = -1.0f;
        }
        else if (p_val > 0.25f)
        {
            p_val = 1.0f;
        }
        else
        {
            p_val = 0.0f;
        }

        if (p_val == y->get_val(i))
        {
            correct_cnt++;
        }
    }

    return correct_cnt;
}

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

    FILE *train_data_file = fopen("temp/train.data", "wb");
    FILE *train_lbl_file = fopen("temp/train.lbl", "wb");

    float data_buf[CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT];
    float lbl_buf;

    int game_cnt = 0;

    for (auto pgn_game : pgn_games)
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
                board.one_hot_encode(data_buf, white);
                lbl_buf = (float)pgn_game->lbl;

                fwrite(data_buf, sizeof(data_buf), 1, train_data_file);
                fwrite(&lbl_buf, sizeof(lbl_buf), 1, train_lbl_file);
            }

            game_move_cnt++;
        }

        game_cnt++;

        if (game_cnt % 1000 == 0)
        {
            printf("Game: %d\n", game_cnt);
        }

        delete pgn_game;
    }

    fclose(train_data_file);
    fclose(train_lbl_file);
}

void train(Model *model, int epochs, int batch_size)
{
    const char *data_path = "temp/train.data";
    const char *lbl_path = "temp/train.lbl";

    int input_size = (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT) * sizeof(float);

    long long data_file_size = FileUtils::get_file_size(data_path);
    size_t data_cnt = data_file_size / input_size;

    int batch_cnt = data_cnt / batch_size;

    FILE *data_file = fopen(data_path, "rb");
    FILE *lbl_file = fopen(lbl_path, "rb");

    // Train:
    {
        FILE *train_csv = fopen("temp/train.csv", "w");
        fprintf(train_csv, "epoch,batch,loss,accuracy\n");

        bool quit = false;

        auto x = Tensor::zeros(false, Shape(batch_size, CHESS_BOARD_CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT));
        auto y = Tensor::zeros(false, Shape(batch_size, 1));

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int batch_idx = 0; batch_idx < batch_cnt; batch_idx++)
            {
                x->to_cpu();
                y->to_cpu();

                fread(x->data(), 1, (input_size * batch_size), data_file);
                fread(y->data(), 1, (sizeof(float) * batch_size), lbl_file);

                auto p = model->forward(x);

                if (batch_idx % 100 == 0)
                {
                    float loss = model->loss(p, y);
                    float acc = model->accuracy(p, y, chess_tanh_accuracy_fn);
                    fprintf(train_csv, "%d,%d,%f,%f\n", epoch, batch_idx, loss, acc);
                }

                model->backward(p, y);
                model->step();

                if (batch_idx == 0)
                {
                    p->print();
                    y->print();
                    for (int i = 0; i < batch_size; i++)
                    {
                        p->set_val(i, p->get_val(i) - y->get_val(i));
                    }
                    p->print();
                }

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

            fseek(data_file, 0, SEEK_SET);
            fseek(lbl_file, 0, SEEK_SET);

            if (quit)
            {
                break;
            }
        }

        delete x;
        delete y;

        fclose(train_csv);
    }

    fclose(data_file);
    fclose(lbl_file);
}

void compare_models(int epochs, int batch_size)
{
    Shape x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT);
    Shape y_shape(batch_size, 1);

    {
        auto model = new Model(new ChessInitializer());

        model->linear(x_shape, 1024, new Tanh());
        model->linear(512, new Tanh());
        model->linear(256, new Tanh());
        model->linear(64, new Tanh());
        model->linear(y_shape, new Tanh());

        model->set_loss(new MSE());
        model->set_optimizer(new ChessOptimizer(model->parameters(), 0.01f, ZERO_NN_BETA_1));

        model->summarize();

        train(model, epochs, batch_size);

        delete model;
    }
}

int main()
{
    srand(time(NULL));

    export_pgn("data/data.pgn");

    // compare_models(10, 128);

    // self_play(3, 3, true);

    return 0;
}