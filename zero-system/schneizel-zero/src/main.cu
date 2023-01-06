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
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 7:
                    if (board_data[square] == CHESS_BN)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 8:
                    if (board_data[square] == CHESS_BB)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 9:
                    if (board_data[square] == CHESS_BR)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 10:
                    if (board_data[square] == CHESS_BQ)
                    {
                        out[out_idx] = 1.0f;
                    }
                    break;
                case 11:
                    if (board_data[square] == CHESS_BK)
                    {
                        out[out_idx] = 1.0f;
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

    char data_buf[CHESS_BOARD_LEN + 1];
    int lbl_buf;

    int game_cnt = 0;

    long long move_cnt = 0;

    for (auto pgn_game : pgn_games)
    {
        Board board;
        bool white = true;

        int game_move_cnt = 0;

        for (auto move_str : pgn_game->move_strs)
        {
            memset(data_buf, 0, sizeof(data_buf));
            memcpy(data_buf, board.get_data(), sizeof(char) * CHESS_BOARD_LEN);

            auto move = board.change(move_str, white);

            if (!Move::is_valid(&move))
            {
                printf("Quitting game %d on move %d\n", game_cnt, game_move_cnt);
                break;
            }

            // Skip openings.
            if (game_move_cnt >= CHESS_OPENING_MOVE_CNT)
            {
                if (white)
                {
                    data_buf[CHESS_BOARD_LEN] = 'w';
                }
                else
                {
                    data_buf[CHESS_BOARD_LEN] = 'b';
                }

                lbl_buf = move.src_square;

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
            printf("Game: %d\tMoves: %ld\n", game_cnt, move_cnt);
        }

        delete pgn_game;
    }

    printf("Game: %d\tMoves: %ld\n", game_cnt, move_cnt);

    fclose(train_data_file);
    fclose(train_lbl_file);
}

int chess_classification_accuracy_fn(Tensor *p, Tensor *y, int batch_size)
{
    int correct_cnt = 0;

    int output_cnt = p->dims_size() / batch_size;

    for (int i = 0; i < batch_size; i++)
    {
        float max_val = y->get_val(i * output_cnt + 0);
        int max_idx = 0;
        for (int j = 1; j < output_cnt; j++)
        {
            float val = y->get_val(i * output_cnt + j);
            if (val > max_val)
            {
                max_val = val;
                max_idx = j;
            }
        }

        if (p->get_val(i * output_cnt + max_idx) >= 0.95f)
        {
            correct_cnt++;
        }
    }

    return correct_cnt;
}

void train_head(int epochs, int batch_size)
{
    Shape x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape y_shape(batch_size, CHESS_BOARD_LEN);

    auto head = new Model(new Xavier());

    head->linear(x_shape, 1024, new ReLU());
    head->linear(1024, new ReLU());
    head->linear(512, new ReLU());
    head->linear(512, new ReLU());
    head->linear(128, new ReLU());
    head->linear(y_shape, new Sigmoid());

    head->set_loss(new CrossEntropy());
    head->set_optimizer(new SGDMomentum(head->parameters(), 0.01f, ZERO_NN_BETA_1));

    head->summarize();

    {
        const char *data_path = "temp/train.data";
        const char *lbl_path = "temp/train.lbl";

        int input_size = CHESS_BOARD_LEN + 1;
        int x_size = (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);

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

            auto x = Tensor::zeros(false, Shape(batch_size, x_size));
            auto y = Tensor::zeros(false, Shape(batch_size, 1));

            char *data_buf = (char *)malloc(sizeof(char) * batch_size * input_size);
            int *lbl_buf = (int *)malloc(sizeof(int) * batch_size);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int batch_idx = 0; batch_idx < batch_cnt; batch_idx++)
                {
                    x->zeros();
                    y->zeros();

                    x->to_cpu();
                    y->to_cpu();

                    fread(data_buf, 1, (input_size * batch_size), data_file);
                    fread(lbl_buf, 1, (sizeof(int) * batch_size), lbl_file);

                    for (int i = 0; i < batch_size; i++)
                    {
                        one_hot_encode_chess_board_data(&data_buf[i * input_size], &x->data()[i * x_size]);
                        if (data_buf[i * input_size + CHESS_BOARD_LEN] == 'w')
                        {
                            x->data()[(i * x_size) + (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT)] = 1.0f;
                        }
                        else
                        {
                            x->data()[(i * x_size) + (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 1.0f;
                        }

                        y->data()[i] = (float)lbl_buf[i];
                    }

                    auto oh_y = Tensor::one_hot(y, CHESS_BOARD_LEN - 1);

                    auto p = head->forward(x);

                    if (batch_idx % 100 == 0)
                    {
                        float loss = head->loss(p, oh_y);
                        float acc = head->accuracy(p, oh_y, chess_classification_accuracy_fn);
                        fprintf(train_csv, "%d,%d,%f,%f\n", epoch, batch_idx, loss, acc);
                    }

                    head->backward(p, oh_y);
                    head->step();

                    if (batch_idx == batch_cnt - 1)
                    {
                        y->print();
                        p->print();
                    }

                    delete p;
                    delete oh_y;

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

            free(data_buf);
            free(lbl_buf);

            fclose(train_csv);
        }

        fclose(data_file);
        fclose(lbl_file);
    }

    head->save_parameters("temp/head.nn");

    delete head;
}

void play(bool white, int depth, Model *head)
{
    Board board;
    Move prev_move;

    OpeningEngine opening_engine("data/openings.data");
    bool opening_stage = true;

    int move_cnt = 0;

    int x_size = (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    auto x = Tensor::zeros(false, Shape(1, x_size));

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

        printf("MATERIAL EVALUATION: %d\n", board.evaluate_material());

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

        if (board.is_check(false))
        {
            printf("WHITE CHECKED!\n");
        }

        if (white)
        {
            do
            {
                std::string move_str;
                printf("ENTER MOVE: ");
                std::cin >> move_str;
                prev_move = board.change(move_str, true);
            } while (!Move::is_valid(&prev_move));
        }
        else
        {
            if (move_cnt == 0)
            {
                if (rand() % 2 == 1)
                {
                    prev_move = board.change("e4", true);
                }
                else
                {
                    prev_move = board.change("d4", true);
                }
            }
            else
            {
                if (opening_stage)
                {
                    std::string move_str = opening_engine.next_move(&board, move_cnt);

                    if (move_str.empty())
                    {
                        printf("\n==================================== END OF BOOK OPENINGS ====================================\n\n");
                        opening_stage = false;
                    }
                    else
                    {
                        prev_move = board.change(move_str, true);
                    }
                }

                if (!opening_stage)
                {
                    auto evals = board.minimax_alphabeta(true, depth, 9, 6);

                    int max_eval_idx = 0;

                    {
                        x->to_cpu();
                        one_hot_encode_chess_board_data(board.get_data(), x->data());
                        x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT)] = 1.0f;
                        x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 0.0f;
                        auto p = head->forward(x);

                        float max_val = 0.0f;

                        auto moves = board.get_all_moves(true);
                        for (auto move : moves)
                        {
                            printf("Src: %d\tDst: %d\n", move.src_square, move.dst_square);
                        }

                        for (int eval_idx = 0; eval_idx < evals.size(); eval_idx++)
                        {
                            auto move = evals[eval_idx].move;

                            float p_val = p->get_val(move.src_square);

                            // Incentivize castling and disincentivize moving king.
                            if (move.src_square == board.get_king_square(true))
                            {
                                int src_dst_diff = abs(move.src_square - move.dst_square);
                                if (src_dst_diff == 2 || src_dst_diff == 3)
                                {
                                    p_val = 20.0f;
                                }
                                else
                                {
                                    p_val = 0.01f;
                                }
                            }

                            // Disincentivize moving knight to rim.
                            if (board.get_piece(move.src_square) == CHESS_WN)
                            {
                                int dst_col = Board::get_col(move.dst_square);

                                if (dst_col == 0 || dst_col == 7)
                                {
                                    p_val = 0.01f;
                                }
                            }

                            printf("Src: %d\tDst: %d\tPiece: %c\tModel: %f\tMaterial: %d\n", move.src_square, move.dst_square, board.get_piece(move.src_square), p_val, evals[eval_idx].value);

                            if (p_val > max_val)
                            {
                                max_eval_idx = eval_idx;
                                max_val = p_val;
                            }
                        }

                        delete p;
                    }

                    if (evals.size() == 0)
                    {
                        printf("ISSUE!!!\n");
                    }

                    board.change(evals[max_eval_idx].move);
                    prev_move = evals[max_eval_idx].move;
                    printf("TIES: %d\n", evals.size());
                }
            }
        }

        move_cnt++;

        printf("\nBLACK TURN\n");
        board.print(prev_move);

        printf("MATERIAL EVALUATION: %d\n", board.evaluate_material());

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

        if (board.is_check(true))
        {
            printf("BLACK CHECKED!\n");
        }

        if (!white)
        {
            do
            {
                std::string move_str;
                printf("ENTER MOVE: ");
                std::cin >> move_str;
                prev_move = board.change(move_str, false);
            } while (!Move::is_valid(&prev_move));
        }
        else
        {
            if (opening_stage)
            {
                std::string move_str = opening_engine.next_move(&board, move_cnt);

                if (move_str.empty())
                {
                    printf("\n==================================== END OF BOOK OPENINGS ====================================\n\n");
                    opening_stage = false;
                }
                else
                {
                    prev_move = board.change(move_str, false);
                }
            }

            if (!opening_stage)
            {
                auto evals = board.minimax_alphabeta(false, depth, 9, 6);

                int max_eval_idx = 0;

                {
                    x->to_cpu();
                    one_hot_encode_chess_board_data(board.get_data(), x->data());
                    x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT)] = 0.0f;
                    x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 1.0f;
                    auto p = head->forward(x);

                    float max_val = 0.0f;

                    auto moves = board.get_all_moves(false);
                    for (auto move : moves)
                    {
                        printf("Src: %d\tDst: %d\n", move.src_square, move.dst_square);
                    }

                    for (int eval_idx = 0; eval_idx < evals.size(); eval_idx++)
                    {
                        auto move = evals[eval_idx].move;

                        float p_val = p->get_val(move.src_square);

                        // Incentivize castling and disincentivize moving king.
                        if (move.src_square == board.get_king_square(false))
                        {
                            int src_dst_diff = abs(move.src_square - move.dst_square);
                            if (src_dst_diff == 2 || src_dst_diff == 3)
                            {
                                p_val = 20.0f;
                            }
                            else
                            {
                                p_val = 0.01f;
                            }
                        }

                        // Disincentivize moving knight to rim.
                        if (board.get_piece(move.src_square) == CHESS_BN)
                        {
                            int dst_col = Board::get_col(move.dst_square);

                            if (dst_col == 0 || dst_col == 7)
                            {
                                p_val = 0.01f;
                            }
                        }

                        printf("Src: %d\tDst: %d\tPiece: %c\tModel: %f\tMaterial: %d\n", move.src_square, move.dst_square, board.get_piece(move.src_square), p_val, evals[eval_idx].value);

                        if (p_val > max_val)
                        {
                            max_eval_idx = eval_idx;
                            max_val = p_val;
                        }
                    }

                    delete p;
                }

                board.change(evals[max_eval_idx].move);
                prev_move = evals[max_eval_idx].move;
                printf("TIES: %d\n", evals.size());
            }
        }

        move_cnt++;
    }

    delete x;
}

int main()
{
    srand(time(NULL));

    Shape x_shape(1, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape y_shape(1, CHESS_BOARD_LEN);

    auto head = new Model(new Xavier());
    {
        head->linear(x_shape, 1024, new ReLU());
        head->linear(1024, new ReLU());
        head->linear(512, new ReLU());
        head->linear(512, new ReLU());
        head->linear(128, new ReLU());
        head->linear(y_shape, new ReLU());

        head->set_loss(new CrossEntropy());
        head->set_optimizer(new SGDMomentum(head->parameters(), 0.001f, ZERO_NN_BETA_1));

        head->load_parameters("data/head.nn");
    }

    play(false, 4, head);

    delete head;

    return 0;
}