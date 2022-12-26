#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

#include "chess.h"

using namespace zero::core;
using namespace zero::nn;

using namespace chess;

#define CHESS_BOARD_CHANNEL_CNT 12

void one_hot_encode_chess_board(Board *board, float *out, bool white)
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
                    if (board->get_piece(square) == CHESS_WP)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 1:
                    if (board->get_piece(square) == CHESS_WN)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 2:
                    if (board->get_piece(square) == CHESS_WB)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 3:
                    if (board->get_piece(square) == CHESS_WR)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 4:
                    if (board->get_piece(square) == CHESS_WQ)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 5:
                    if (board->get_piece(square) == CHESS_WK)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 6:
                    if (board->get_piece(square) == CHESS_BP)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 7:
                    if (board->get_piece(square) == CHESS_BN)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 8:
                    if (board->get_piece(square) == CHESS_BB)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 9:
                    if (board->get_piece(square) == CHESS_BR)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 10:
                    if (board->get_piece(square) == CHESS_BQ)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 11:
                    if (board->get_piece(square) == CHESS_BK)
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

        if (board.is_checkmate(false))
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
        game.boards.push_back(board);

        move_cnt++;

        if (print)
        {
            printf("\nBLACK TURN\n");
            board.print(prev_move);
        }

        if (board.is_checkmate(true))
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
        game.boards.push_back(board);

        move_cnt++;
    }

    return game;
}

void play(bool white, int depth)
{
    Board board;
    Move prev_move;

    OpeningEngine opening_engine;
    bool opening_stage = true;

    int move_cnt = 0;

    while (true)
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

        if (board.is_checkmate(false))
        {
            printf("WHITE CHECKMATED!\n");
            break;
        }
        else if (!board.has_moves(true))
        {
            printf("WHITE STALEMATED!\n");
            break;
        }

        if (white)
        {
            std::string move_str;
            printf("Enter Move: ");
            std::cin >> move_str;
            prev_move = board.change(move_str, true);
        }
        else
        {
            if (move_cnt == 0)
            {
                prev_move = board.change("e4", true);
            }
            else
            {
                if (opening_stage && !opening_engine.matches(&board, move_cnt))
                {
                    printf("End of book opening\n");
                    opening_stage = false;
                }

                if (!opening_stage)
                {
                    auto evals = board.minimax_alphabeta_dyn(true, depth);
                    int r = rand() % evals.size();
                    board.change(evals[r].move);
                    prev_move = evals[r].move;
                }
            }
        }

        move_cnt++;

        printf("\nBLACK TURN\n");
        board.print(prev_move);

        if (board.is_checkmate(true))
        {
            printf("BLACK CHECKMATED!\n");
            break;
        }
        else if (!board.has_moves(false))
        {
            printf("BLACK STALEMATED!\n");
            break;
        }

        if (!white)
        {
            std::string move_str;
            printf("Enter Move: ");
            std::cin >> move_str;
            prev_move = board.change(move_str, false);
        }
        else
        {
            if (opening_stage && !opening_engine.matches(&board, move_cnt))
            {
                printf("End of book opening\n");
                opening_stage = false;
            }

            if (!opening_stage)
            {
                auto evals = board.minimax_alphabeta_dyn(false, depth);
                int r = rand() % evals.size();
                board.change(evals[r].move);
                prev_move = evals[r].move;
            }
        }

        move_cnt++;
    }
}

void play_no_opening_engine(bool white, int depth)
{
    Board board;
    Move prev_move;

    int move_cnt = 0;

    while (true)
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

        if (board.is_checkmate(false))
        {
            printf("WHITE CHECKMATED!\n");
            break;
        }
        else if (!board.has_moves(true))
        {
            printf("WHITE STALEMATED!\n");
            break;
        }

        if (white)
        {
            std::string move_str;
            printf("Enter Move: ");
            std::cin >> move_str;
            prev_move = board.change(move_str, true);
        }
        else
        {
            if (move_cnt == 0)
            {
                prev_move = board.change("e4", true);
            }
            else
            {
                auto evals = board.minimax_alphabeta_dyn(true, depth);
                int r = rand() % evals.size();
                board.change(evals[r].move);
                prev_move = evals[r].move;
            }
        }

        move_cnt++;

        printf("\nBLACK TURN\n");
        board.print(prev_move);

        if (board.is_checkmate(true))
        {
            printf("BLACK CHECKMATED!\n");
            break;
        }
        else if (!board.has_moves(false))
        {
            printf("BLACK STALEMATED!\n");
            break;
        }

        if (!white)
        {
            std::string move_str;
            printf("Enter Move: ");
            std::cin >> move_str;
            prev_move = board.change(move_str, false);
        }
        else
        {
            auto evals = board.minimax_alphabeta_dyn(false, depth);
            int r = rand() % evals.size();
            board.change(evals[r].move);
            prev_move = evals[r].move;
        }

        move_cnt++;
    }
}

void export_pgn(const char *path)
{
    auto pgn_games = PGN::import(path, FileUtils::get_file_size(path));

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
                one_hot_encode_chess_board(&board, data_buf, white);
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
                    float acc = model->accuracy(p, y, Model::classification_accuracy_fn);
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
        auto model = new Model(new Xavier());

        model->linear(x_shape, 1024, new Tanh());
        model->linear(512, new Tanh());
        model->linear(256, new Tanh());
        model->linear(64, new Tanh());
        model->linear(y_shape, new Tanh());

        model->set_loss(new MSE());
        model->set_optimizer(new SGDMomentum(model->parameters(), 0.01f, ZERO_NN_BETA_1));

        model->summarize();

        train(model, epochs, batch_size);

        delete model;
    }
}

int main()
{
    srand(time(NULL));

    // const char *path = "C:\\dev\\ml-portfolio\\zero-system\\schneizel-zero\\data\\data.pgn";

    // export_pgn(path);

    // compare_models(10, 128);

    // self_play(3, 3, true);

    // play_no_opening_engine(true, 3);

    play(false, 5);

    return 0;
}