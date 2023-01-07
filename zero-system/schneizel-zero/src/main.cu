#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

#include "chess.h"

using namespace zero::core;
using namespace zero::nn;

using namespace chess;

#define CHESS_BOARD_CHANNEL_CNT 12

void one_hot_encode_chess_board_data(const char *board_data, float *out)
{
    memset(out, 0, sizeof(float) * CHESS_BOARD_CHANNEL_CNT * CHESS_BOARD_LEN);
    for (int c = 0; c < CHESS_BOARD_CHANNEL_CNT; c++)
    {
        for (int i = 0; i < CHESS_ROW_CNT; i++)
        {
            for (int j = 0; j < CHESS_COL_CNT; j++)
            {
                int channel_offset = (c * CHESS_BOARD_LEN);
                int square = (i * CHESS_COL_CNT) + j;
                int out_idx = channel_offset + square;

                switch (c)
                {
                case 0:
                    if (board_data[square] == CHESS_WP)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 1:
                    if (board_data[square] == CHESS_WN)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 2:
                    if (board_data[square] == CHESS_WB)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 3:
                    if (board_data[square] == CHESS_WR)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 4:
                    if (board_data[square] == CHESS_WQ)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 5:
                    if (board_data[square] == CHESS_WK)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 6:
                    if (board_data[square] == CHESS_BP)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 7:
                    if (board_data[square] == CHESS_BN)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 8:
                    if (board_data[square] == CHESS_BB)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 9:
                    if (board_data[square] == CHESS_BR)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 10:
                    if (board_data[square] == CHESS_BQ)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 11:
                    if (board_data[square] == CHESS_BK)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                default:
                    break;
                }
            }
        }
    }
}

void export_pgn(const char *path)
{
    auto pgn_games = PGN::import(path, FileUtils::get_file_size(path));

    FILE *train_data_file = fopen("temp/train.data", "wb");
    FILE *train_lbl_file = fopen("temp/train.lbl", "wb");

    char data_buf[CHESS_BOARD_LEN];
    int lbl_buf;

    int game_cnt = 0;

    long long move_cnt = 0;

    for (auto pgn_game : pgn_games)
    {
        Board board;
        bool white = true;

        int game_move_cnt = 0;

        lbl_buf = pgn_game->lbl;

        for (auto move_str : pgn_game->move_strs)
        {
            auto move = board.change(move_str, white);

            if (!Move::is_valid(&move))
            {
                printf("Quitting game %d on move %d\n", game_cnt, game_move_cnt);
                break;
            }

            // Skip openings.
            if (game_move_cnt >= CHESS_OPENING_MOVE_CNT)
            {
                memcpy(data_buf, board.get_data(), sizeof(char) * CHESS_BOARD_LEN);

                fwrite(data_buf, sizeof(data_buf), 1, train_data_file);
                fwrite(&lbl_buf, sizeof(lbl_buf), 1, train_lbl_file);

                move_cnt++;
            }

            white = !white;

            game_move_cnt++;
        }

        game_cnt++;

        if (game_cnt % 1000 == 0)
        {
            printf("Games: %d\tMoves: %ld\n", game_cnt, move_cnt);
        }

        delete pgn_game;
    }

    printf("Games: %d\tMoves: %ld\n", game_cnt, move_cnt);

    fclose(train_data_file);
    fclose(train_lbl_file);
}

int chess_accuracy_fn(Tensor *p, Tensor *y, int batch_size)
{
    int correct_cnt = 0;

    int output_cnt = p->dims_size() / batch_size;

    for (int i = 0; i < batch_size; i++)
    {
        float y_val = y->get_val(i);
        float p_val = p->get_val(i);

        if (y_val > 0.0f)
        {
            // White win.
            if (p_val >= 0.25f)
            {
                correct_cnt++;
            }
        }
        else if (y_val < 0.0f)
        {
            // Black win.
            if (p_val <= -0.25f)
            {
                correct_cnt++;
            }
        }
        else
        {
            // Draw.
            if (p_val < 0.25f && p_val > -0.25f)
            {
                correct_cnt++;
            }
        }
    }

    return correct_cnt;
}

Model *get_model(int batch_size, const char *params_path)
{
    Shape x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT);
    Shape y_shape(batch_size, 1);

    auto model = new Model(new Xavier());

    model->hadamard_product(x_shape, 16, new Tanh());
    model->hadamard_product(16, new Tanh());
    model->matrix_product(16, new Tanh());
    model->matrix_product(16, new Tanh());
    model->linear(512, new Tanh());
    model->linear(128, new Tanh());
    model->linear(32, new Tanh());
    model->linear(y_shape, new Tanh());

    model->set_loss(new MSE());
    model->set_optimizer(new SGDMomentum(model->parameters(), 0.01f, ZERO_NN_BETA_1));

    model->summarize();

    if (params_path != nullptr)
    {
        model->load_parameters(params_path);
    }

    return model;
}

void train(int epochs, int batch_size)
{
    Shape x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT);
    Shape y_shape(batch_size, 1);

    auto model = get_model(batch_size, nullptr);

    {
        const char *data_path = "temp/train.data";
        const char *lbl_path = "temp/train.lbl";

        int data_size = CHESS_BOARD_LEN;
        int x_size = (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT);

        long long data_file_size = FileUtils::get_file_size(data_path);
        size_t data_cnt = data_file_size / data_size;

        int batch_cnt = data_cnt / batch_size;

        FILE *data_file = fopen(data_path, "rb");
        FILE *lbl_file = fopen(lbl_path, "rb");

        {
            FILE *train_csv = fopen("temp/train.csv", "w");
            fprintf(train_csv, "epoch,batch,loss,accuracy\n");

            bool quit = false;

            auto x = Tensor::zeros(false, x_shape);
            auto y = Tensor::zeros(false, y_shape);

            char data_buf[CHESS_BOARD_LEN];
            int lbl_buf;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int batch_idx = 0; batch_idx < batch_cnt; batch_idx++)
                {
                    x->zeros();
                    y->zeros();

                    x->to_cpu();
                    y->to_cpu();

                    for (int i = 0; i < batch_size; i++)
                    {
                        long long offset = rand() % data_cnt;
                        fseek(data_file, offset * data_size, SEEK_SET);
                        fseek(lbl_file, offset * sizeof(int), SEEK_SET);

                        fread(data_buf, data_size, 1, data_file);
                        fread(&lbl_buf, sizeof(int), 1, lbl_file);

                        one_hot_encode_chess_board_data(data_buf, &x->data()[i * x_size]);
                        y->data()[i] = (float)lbl_buf;
                    }

                    auto p = model->forward(x);

                    if (batch_idx % 100 == 0)
                    {
                        float loss = model->loss(p, y);
                        float acc = model->accuracy(p, y, chess_accuracy_fn);
                        fprintf(train_csv, "%d,%d,%f,%f\n", epoch, batch_idx, loss, acc);
                    }

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

                    fseek(data_file, 0, SEEK_SET);
                    fseek(lbl_file, 0, SEEK_SET);
                }

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

    model->save_parameters("temp/model.nn");

    delete model;
}

int main()
{
    srand(time(NULL));

    // export_pgn("data/all.pgn");

    train(10, 64);

    return 0;
}