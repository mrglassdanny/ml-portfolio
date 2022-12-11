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

void self_play_test1()
{
    auto x = Tensor::zeros(false, Shape(1, 6, 8, 8));
    auto y = Tensor::zeros(true, Shape(1, 1));

    auto model = new Model();
    model->hadamard_product(x->shape(), 16, layer::ActivationType::Tanh);
    model->matrix_product(16, layer::ActivationType::Tanh);
    model->linear(y->shape(), layer::ActivationType::Tanh);

    model->set_loss(new loss::MSE());
    model->set_optimizer(new optim::SGD(model->parameters(), 0.01f));

    model->summarize();

    int game_cnt = 0;

    while (true)
    {
        game_cnt++;
        auto game = self_play(3, 3, false, model);
        if (game.lbl != 0)
        {
            for (auto board : game.boards)
            {
                x->to_cpu();
                board.one_hot_encode(x->data());
                x->to_cuda();
                y->set_val(0, game.lbl);

                board.print();

                auto p = model->forward(x);
                printf("\tLOSS: %f\n", model->loss(p, y));
                p->print();
                y->print();
                model->backward(p, y);
                model->step();

                delete p;
            }
        }

        printf("GAME COUNT: %d\n", game_cnt);
    }

    delete x;
    delete y;
}

void export_pgn(const char *path)
{
    auto pgn_games = PGN::import(path);

    FILE *train_data_file = fopen("temp/train.data", "wb");
    FILE *train_lbl_file = fopen("temp/train.lbl", "wb");
    FILE *test_data_file = fopen("temp/test.data", "wb");
    FILE *test_lbl_file = fopen("temp/test.lbl", "wb");

    float data_buf[6 * 8 * 8];
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
        }

        delete pgn_game;
    }

    fclose(train_data_file);
    fclose(train_lbl_file);
    fclose(test_data_file);
    fclose(test_lbl_file);
}

struct Batch
{
    zero::core::Tensor *x;
    zero::core::Tensor *y;
};

std::vector<Batch> get_train_dataset(int batch_size)
{
    int oh_board_len = 6 * 8 * 8;
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
        auto x = zero::core::Tensor::from_data(zero::core::Shape(batch_size, 6, 8, 8), &data_buf[i * batch_size * oh_board_len]);
        auto y = zero::core::Tensor::from_data(zero::core::Shape(batch_size, 1), &lbl_buf[i * batch_size]);

        batches.push_back({x, y});
    }

    free(data_buf);
    free(lbl_buf);

    return batches;
}

std::vector<Batch> get_test_dataset(int batch_size)
{
    int oh_board_len = 6 * 8 * 8;
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
        auto x = zero::core::Tensor::from_data(zero::core::Shape(batch_size, 6, 8, 8), &data_buf[i * batch_size * oh_board_len]);
        auto y = zero::core::Tensor::from_data(zero::core::Shape(batch_size, 1), &lbl_buf[i * batch_size]);

        batches.push_back({x, y});
    }

    free(data_buf);
    free(lbl_buf);

    return batches;
}

int main()
{
    srand(time(NULL));

    // export_pgn("data/data.pgn");

    auto tr = get_train_dataset(128);
    auto te = get_test_dataset(128);


    return 0;
}