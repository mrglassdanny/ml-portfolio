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
    FILE *train_head_lbl_file = fopen("temp/train-head.lbl", "wb");
    FILE *train_hand_lbl_file = fopen("temp/train-hand.lbl", "wb");

    char data_buf[CHESS_BOARD_LEN + 1];
    int head_lbl_buf;
    int hand_lbl_buf;

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

                head_lbl_buf = move.src_square;
                hand_lbl_buf = move.dst_square;

                fwrite(data_buf, sizeof(data_buf), 1, train_data_file);
                fwrite(&head_lbl_buf, sizeof(head_lbl_buf), 1, train_head_lbl_file);
                fwrite(&hand_lbl_buf, sizeof(hand_lbl_buf), 1, train_hand_lbl_file);

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
    fclose(train_head_lbl_file);
    fclose(train_hand_lbl_file);
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

void train(int epochs, int batch_size)
{
    Shape head_x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape hand_x_shape(batch_size, (CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape y_shape(batch_size, CHESS_BOARD_LEN);

    // auto head = new Model(new Xavier());
    // {
    //     head->linear(head_x_shape, 1024, new ReLU());
    //     head->linear(1024, new ReLU());
    //     head->linear(512, new ReLU());
    //     head->linear(512, new ReLU());
    //     head->linear(128, new ReLU());
    //     head->linear(y_shape, new Sigmoid());

    //     head->set_loss(new CrossEntropy());
    //     head->set_optimizer(new SGDMomentum(head->parameters(), 0.01f, ZERO_NN_BETA_1));

    //     head->summarize();
    // }

    // auto hand = new Model(new Xavier());
    // {
    //     hand->linear(hand_x_shape, 1024, new ReLU());
    //     hand->linear(1024, new ReLU());
    //     hand->linear(512, new ReLU());
    //     hand->linear(512, new ReLU());
    //     hand->linear(128, new ReLU());
    //     hand->linear(y_shape, new Sigmoid());

    //     hand->set_loss(new CrossEntropy());
    //     hand->set_optimizer(new SGDMomentum(hand->parameters(), 0.01f, ZERO_NN_BETA_1));

    //     hand->summarize();
    // }

    auto head = new Model(new Xavier());
    {
        head->linear(head_x_shape, 512, new ReLU());
        head->linear(256, new ReLU());
        head->linear(128, new ReLU());
        head->linear(y_shape, new Sigmoid());

        head->set_loss(new CrossEntropy());
        head->set_optimizer(new SGDMomentum(head->parameters(), 0.1f, ZERO_NN_BETA_1));

        head->summarize();
    }

    auto hand = new Model(new Xavier());
    {
        hand->linear(hand_x_shape, 512, new ReLU());
        hand->linear(256, new ReLU());
        hand->linear(128, new ReLU());
        hand->linear(y_shape, new Sigmoid());

        hand->set_loss(new CrossEntropy());
        hand->set_optimizer(new SGDMomentum(hand->parameters(), 0.1f, ZERO_NN_BETA_1));

        hand->summarize();
    }

    {
        const char *data_path = "temp/train.data";
        const char *head_lbl_path = "temp/train-head.lbl";
        const char *hand_lbl_path = "temp/train-hand.lbl";

        int input_size = CHESS_BOARD_LEN + 1;
        int head_x_size = (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
        int hand_x_size = ((CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT + 2);

        long long data_file_size = FileUtils::get_file_size(data_path);
        size_t data_cnt = data_file_size / input_size;

        int batch_cnt = data_cnt / batch_size;

        FILE *data_file = fopen(data_path, "rb");
        FILE *head_lbl_file = fopen(head_lbl_path, "rb");
        FILE *hand_lbl_file = fopen(hand_lbl_path, "rb");

        {
            FILE *train_head_csv = fopen("temp/train-head.csv", "w");
            fprintf(train_head_csv, "epoch,batch,loss,accuracy\n");

            FILE *train_hand_csv = fopen("temp/train-hand.csv", "w");
            fprintf(train_hand_csv, "epoch,batch,loss,accuracy\n");

            bool quit = false;

            auto head_x = Tensor::zeros(false, head_x_shape);
            auto hand_x = Tensor::zeros(false, hand_x_shape);
            auto head_y = Tensor::zeros(false, Shape(batch_size, 1));
            auto hand_y = Tensor::zeros(false, Shape(batch_size, 1));

            char *data_buf = (char *)malloc(sizeof(char) * batch_size * input_size);
            int *head_lbl_buf = (int *)malloc(sizeof(int) * batch_size);
            int *hand_lbl_buf = (int *)malloc(sizeof(int) * batch_size);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int batch_idx = 0; batch_idx < batch_cnt; batch_idx++)
                {
                    head_x->zeros();
                    head_y->zeros();

                    hand_x->zeros();
                    hand_y->zeros();

                    head_x->to_cpu();
                    head_y->to_cpu();

                    hand_x->to_cpu();
                    hand_y->to_cpu();

                    fread(data_buf, 1, (input_size * batch_size), data_file);
                    fread(head_lbl_buf, 1, (sizeof(int) * batch_size), head_lbl_file);
                    fread(hand_lbl_buf, 1, (sizeof(int) * batch_size), hand_lbl_file);

                    for (int i = 0; i < batch_size; i++)
                    {
                        one_hot_encode_chess_board_data(&data_buf[i * input_size], &head_x->data()[i * head_x_size]);
                        memcpy(&hand_x->data()[(i * hand_x_size)], &head_x->data()[i * head_x_size], sizeof(float) * CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT);

                        if (data_buf[i * input_size + CHESS_BOARD_LEN] == 'w')
                        {
                            head_x->data()[(i * head_x_size) + (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT)] = 1.0f;
                            hand_x->data()[(i * hand_x_size) + ((CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT)] = 1.0f;
                        }
                        else
                        {
                            head_x->data()[(i * head_x_size) + (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 1.0f;
                            hand_x->data()[(i * hand_x_size) + ((CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 1.0f;
                        }

                        head_y->data()[i] = (float)head_lbl_buf[i];
                        hand_y->data()[i] = (float)hand_lbl_buf[i];
                    }

                    auto head_oh_y = Tensor::one_hot(head_y, CHESS_BOARD_LEN - 1);
                    auto hand_oh_y = Tensor::one_hot(hand_y, CHESS_BOARD_LEN - 1);

                    for (int i = 0; i < batch_size; i++)
                    {
                        memcpy(&hand_x->data()[(i * hand_x_size) + (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT)],
                               &head_oh_y->data()[i * CHESS_BOARD_LEN], sizeof(float) * CHESS_ROW_CNT * CHESS_COL_CNT);
                    }

                    auto head_p = head->forward(head_x);
                    auto hand_p = hand->forward(hand_x);

                    if (batch_idx % 100 == 0)
                    {
                        float head_loss = head->loss(head_p, head_oh_y);
                        float head_acc = head->accuracy(head_p, head_oh_y, chess_classification_accuracy_fn);
                        fprintf(train_head_csv, "%d,%d,%f,%f\n", epoch, batch_idx, head_loss, head_acc);

                        float hand_loss = hand->loss(hand_p, hand_oh_y);
                        float hand_acc = hand->accuracy(hand_p, hand_oh_y, chess_classification_accuracy_fn);
                        fprintf(train_hand_csv, "%d,%d,%f,%f\n", epoch, batch_idx, hand_loss, hand_acc);
                    }

                    head->backward(head_p, head_oh_y);
                    hand->backward(hand_p, hand_oh_y);

                    head->step();
                    hand->step();

                    delete head_p;
                    delete head_oh_y;
                    delete hand_p;
                    delete hand_oh_y;

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
                fseek(head_lbl_file, 0, SEEK_SET);
                fseek(hand_lbl_file, 0, SEEK_SET);

                if (quit)
                {
                    break;
                }
            }

            delete head_x;
            delete head_y;

            delete hand_x;
            delete hand_y;

            free(data_buf);
            free(head_lbl_buf);
            free(hand_lbl_buf);

            fclose(train_head_csv);
            fclose(train_hand_csv);
        }

        fclose(data_file);
        fclose(head_lbl_file);
        fclose(hand_lbl_file);
    }

    head->save_parameters("temp/head.nn");
    hand->save_parameters("temp/hand.nn");

    delete head;
    delete hand;
}

void play(bool white, int depth, Model *head, Model *hand)
{
    Board board;
    Move prev_move;

    OpeningEngine opening_engine("data/openings.data");
    bool opening_stage = true;

    int move_cnt = 0;

    Shape head_x_shape(1, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape hand_x_shape(1, (CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT + 2);

    auto head_x = Tensor::zeros(false, head_x_shape);
    auto hand_x = Tensor::zeros(false, hand_x_shape);

    float hand_move_buf[CHESS_BOARD_LEN];

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
                    auto eval_dataset = board.minimax_alphabeta(true, depth, 9, 6);

                    int max_eval_idx = 0;

                    {
                        head_x->to_cpu();
                        hand_x->to_cpu();

                        one_hot_encode_chess_board_data(board.get_data(), head_x->data());
                        memcpy(hand_x->data(), head_x->data(), sizeof(float) * CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT);

                        head_x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT)] = 1.0f;
                        hand_x->data()[((CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT)] = 1.0f;

                        head_x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 0.0f;
                        hand_x->data()[((CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 0.0f;

                        auto head_p = head->forward(head_x);

                        float max_head_val = -FLT_MAX;
                        float max_hand_val = -FLT_MAX;

                        for (int eval_data_idx = 0; eval_data_idx < eval_dataset.size(); eval_data_idx++)
                        {
                            auto move = eval_dataset[eval_data_idx].move;

                            hand_x->to_cpu();
                            memset(hand_move_buf, 0, sizeof(hand_move_buf));
                            hand_move_buf[move.src_square] = 1.0f;
                            memcpy(&hand_x->data()[CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT], hand_move_buf, sizeof(hand_move_buf));

                            auto hand_p = hand->forward(hand_x);

                            float head_p_val = head_p->get_val(move.src_square);
                            float hand_p_val = hand_p->get_val(move.dst_square);

                            delete hand_p;

                            // Incentivize castling and disincentivize moving king.
                            if (move.src_square == board.get_king_square(true))
                            {
                                int src_dst_diff = abs(move.src_square - move.dst_square);
                                if (src_dst_diff == 2 || src_dst_diff == 3)
                                {
                                    head_p_val = 20.0f;
                                }
                                else
                                {
                                    head_p_val = 0.01f;
                                }
                            }

                            // Disincentivize moving knight to rim.
                            if (board.get_piece(move.src_square) == CHESS_WN)
                            {
                                int dst_col = Board::get_col(move.dst_square);

                                if (dst_col == 0 || dst_col == 7)
                                {
                                    head_p_val = 0.01f;
                                }
                            }

                            printf("Src: %d\tDst: %d\tPiece: %c\tHead: %f\tHand: %f\tAvg: %f\tMaterial: %d\tDepth: %d\n", move.src_square, move.dst_square, board.get_piece(move.src_square), head_p_val, hand_p_val, (head_p_val + hand_p_val) / 2.0f, eval_dataset[eval_data_idx].eval.value, eval_dataset[eval_data_idx].eval.depth);

                            if (head_p_val > max_head_val)
                            {
                                max_eval_idx = eval_data_idx;
                                max_head_val = head_p_val;
                                max_hand_val = hand_p_val;
                            }
                            else if (head_p_val == max_head_val)
                            {
                                if (hand_p_val > max_hand_val)
                                {
                                    max_eval_idx = eval_data_idx;
                                    max_hand_val = hand_p_val;
                                }
                            }
                        }

                        delete head_p;
                    }

                    board.change(eval_dataset[max_eval_idx].move);
                    prev_move = eval_dataset[max_eval_idx].move;
                    printf("TIES: %d\n", eval_dataset.size());
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
                auto eval_dataset = board.minimax_alphabeta(false, depth, 9, 6);

                int max_eval_idx = 0;

                {
                    head_x->to_cpu();
                    hand_x->to_cpu();

                    one_hot_encode_chess_board_data(board.get_data(), head_x->data());
                    memcpy(hand_x->data(), head_x->data(), sizeof(float) * CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT);

                    head_x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT)] = 0.0f;
                    hand_x->data()[((CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT)] = 0.0f;

                    head_x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 1.0f;
                    hand_x->data()[((CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 1.0f;

                    auto head_p = head->forward(head_x);

                    float max_head_val = -FLT_MAX;
                    float max_hand_val = -FLT_MAX;

                    for (int eval_data_idx = 0; eval_data_idx < eval_dataset.size(); eval_data_idx++)
                    {
                        auto move = eval_dataset[eval_data_idx].move;

                        hand_x->to_cpu();

                        memset(hand_move_buf, 0, sizeof(hand_move_buf));
                        hand_move_buf[move.src_square] = 1.0f;
                        memcpy(&hand_x->data()[CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT], hand_move_buf, sizeof(hand_move_buf));

                        auto hand_p = hand->forward(hand_x);

                        float head_p_val = head_p->get_val(move.src_square);
                        float hand_p_val = hand_p->get_val(move.dst_square);

                        delete hand_p;

                        // Incentivize castling and disincentivize moving king.
                        if (move.src_square == board.get_king_square(false))
                        {
                            int src_dst_diff = abs(move.src_square - move.dst_square);
                            if (src_dst_diff == 2 || src_dst_diff == 3)
                            {
                                head_p_val = 20.0f;
                            }
                            else
                            {
                                head_p_val = 0.01f;
                            }
                        }

                        // Disincentivize moving knight to rim.
                        if (board.get_piece(move.src_square) == CHESS_BN)
                        {
                            int dst_col = Board::get_col(move.dst_square);

                            if (dst_col == 0 || dst_col == 7)
                            {
                                head_p_val = 0.01f;
                            }
                        }

                        printf("Src: %d\tDst: %d\tPiece: %c\tHead: %f\tHand: %f\tAvg: %f\tMaterial: %d\tDepth: %d\n", move.src_square, move.dst_square, board.get_piece(move.src_square), head_p_val, hand_p_val, (head_p_val + hand_p_val) / 2.0f, eval_dataset[eval_data_idx].eval.value, eval_dataset[eval_data_idx].eval.depth);

                        if (head_p_val > max_head_val)
                        {
                            max_eval_idx = eval_data_idx;
                            max_head_val = head_p_val;
                            max_hand_val = hand_p_val;
                        }
                        else if (head_p_val == max_head_val)
                        {
                            if (hand_p_val > max_hand_val)
                            {
                                max_eval_idx = eval_data_idx;
                                max_hand_val = hand_p_val;
                            }
                        }
                    }

                    delete head_p;
                }

                board.change(eval_dataset[max_eval_idx].move);
                prev_move = eval_dataset[max_eval_idx].move;
                printf("TIES: %d\n", eval_dataset.size());
            }
        }

        move_cnt++;
    }

    delete head_x;
    delete hand_x;
}

int main()
{
    srand(time(NULL));

    // Shape x_shape(1, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    // Shape y_shape(1, CHESS_BOARD_LEN);

    // auto head = new Model(new Xavier());
    // {
    //     head->linear(x_shape, 1024, new ReLU());
    //     head->linear(1024, new ReLU());
    //     head->linear(512, new ReLU());
    //     head->linear(512, new ReLU());
    //     head->linear(128, new ReLU());
    //     head->linear(y_shape, new ReLU());

    //     head->set_loss(new CrossEntropy());
    //     head->set_optimizer(new SGDMomentum(head->parameters(), 0.001f, ZERO_NN_BETA_1));

    //     head->load_parameters("data/head.nn");
    // }

    // delete head;

    // export_pgn("data/all.pgn");

    // train(10, 128);

    Shape head_x_shape(1, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape hand_x_shape(1, (CHESS_BOARD_CHANNEL_CNT + 1) * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape y_shape(1, CHESS_BOARD_LEN);

    auto head = new Model(new Xavier());
    {
        head->linear(head_x_shape, 512, new ReLU());
        head->linear(256, new ReLU());
        head->linear(128, new ReLU());
        head->linear(y_shape, new ReLU());

        head->set_loss(new CrossEntropy());
        head->set_optimizer(new SGDMomentum(head->parameters(), 0.1f, ZERO_NN_BETA_1));

        head->load_parameters("data/head.nn");
    }

    auto hand = new Model(new Xavier());
    {
        hand->linear(hand_x_shape, 512, new ReLU());
        hand->linear(256, new ReLU());
        hand->linear(128, new ReLU());
        hand->linear(y_shape, new ReLU());

        hand->set_loss(new CrossEntropy());
        hand->set_optimizer(new SGDMomentum(hand->parameters(), 0.1f, ZERO_NN_BETA_1));

        hand->load_parameters("data/hand.nn");
    }

    play(true, 5, head, hand);

    return 0;
}