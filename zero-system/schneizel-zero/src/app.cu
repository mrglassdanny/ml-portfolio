#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

#include "chess.cuh"

using namespace zero::core;
using namespace zero::nn;
using namespace chess;

struct Game
{
    std::vector<Board> boards;
    float lbl;
};

Game self_play(int white_depth, int black_depth, bool print, Model *model)
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
            printf("\nWHITE TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
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

        prev_move = board.change_minimax_async(true, white_depth, model);
        Board cpy_board;
        cpy_board.copy(&board);
        game.boards.push_back(cpy_board);

        move_cnt++;

        if (print)
        {
            printf("\nBLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
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

        prev_move = board.change_minimax_async(false, black_depth);
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

    for (auto pgn_game : pgn_games)
    {
        Board board;
        bool white = true;

        for (auto move_str : pgn_game->move_strs)
        {
            auto move = board.change(move_str, white);
            white = !white;

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

        game_cnt++;
        delete pgn_game;
    }

    fclose(train_data_file);
    fclose(train_lbl_file);
    fclose(test_data_file);
    fclose(test_lbl_file);

    printf("GAME COUNT: %d\tMOVE COUNT: %ld", game_cnt, move_cnt);
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
            acc += model->accuracy(p, y);
            delete p;
        }

        float test_acc_pct = (acc / (float)test_batch_cnt) * 100.0f;

        model->summarize();
        printf("TEST LOSS: %f\tTEST ACCURACY: %f%%\n",
               (loss / (float)test_batch_cnt),
               test_acc_pct);
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
        auto model = new Model();
        model->hadamard_product(x_shape, 16, ActivationType::Tanh);
        model->matrix_product(16, ActivationType::Tanh);
        model->linear(y_shape, ActivationType::Tanh);
        model->set_loss(new MSE());
        model->set_optimizer(new SGD(model->parameters(), 0.1f));

        train_n_test(model, epochs, &train_ds, &test_ds);

        delete model;
    }

    {
        printf("\n\n");
        auto model = new Model();
        model->hadamard_product(x_shape, 64, ActivationType::Tanh);
        model->matrix_product(64, ActivationType::Tanh);
        model->linear(y_shape, ActivationType::Tanh);
        model->set_loss(new MSE());
        model->set_optimizer(new SGD(model->parameters(), 0.1f));

        train_n_test(model, epochs, &train_ds, &test_ds);

        delete model;
    }

    {
        printf("\n\n");
        auto model = new Model();
        model->hadamard_product(x_shape, 16, ActivationType::Tanh);
        model->hadamard_product(16, ActivationType::Tanh);
        model->matrix_product(16, ActivationType::Tanh);
        model->matrix_product(16, ActivationType::Tanh);
        model->linear(y_shape, ActivationType::Tanh);
        model->set_loss(new MSE());
        model->set_optimizer(new SGD(model->parameters(), 0.1f));

        train_n_test(model, epochs, &train_ds, &test_ds);

        delete model;
    }

    {
        printf("\n\n");
        auto model = new Model();
        model->hadamard_product(x_shape, 64, ActivationType::Tanh);
        model->linear(512, ActivationType::Tanh);
        model->linear(y_shape, ActivationType::Tanh);
        model->set_loss(new MSE());
        model->set_optimizer(new SGD(model->parameters(), 0.1f));

        train_n_test(model, epochs, &train_ds, &test_ds);

        delete model;
    }

    // Clean up:

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

    export_pgn("data/data.pgn");

    // compare_models(4);

    return 0;
}