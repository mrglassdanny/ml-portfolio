#include <stdio.h>
#include <conio.h>

#include <map>

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

int schneizel_sim_minimax_alphabeta_sync(Simulation sim, bool white, int depth, int depth_inc_cnt, int depth_inc_max_move_cnt, int alpha, int beta)
{
    if (sim.board.is_checkmate(!white, false))
    {
        if (white)
        {
            return CHESS_EVAL_MAX_VAL;
        }
        else
        {
            return CHESS_EVAL_MIN_VAL;
        }
    }

    if (depth == 0)
    {
        return sim.board.evaluate_material();
    }

    if (!white)
    {
        int best_eval_val = CHESS_EVAL_MIN_VAL;
        auto sim_sims = sim.board.simulate_all(true);

        if (sim_sims.size() <= depth_inc_max_move_cnt && depth_inc_cnt > 0)
        {
            depth++;
            depth_inc_cnt--;
        }

        for (auto sim_sim : sim_sims)
        {
            int eval_val = schneizel_sim_minimax_alphabeta_sync(sim_sim, true, depth - 1, depth_inc_cnt, depth_inc_max_move_cnt, alpha, beta);

            best_eval_val = eval_val > best_eval_val ? eval_val : best_eval_val;

            alpha = eval_val > alpha ? eval_val : alpha;
            if (beta <= alpha)
            {
                break;
            }
        }

        return best_eval_val;
    }
    else
    {
        int best_eval_val = CHESS_EVAL_MAX_VAL;
        auto sim_sims = sim.board.simulate_all(false);

        if (sim_sims.size() <= depth_inc_max_move_cnt && depth_inc_cnt > 0)
        {
            depth++;
            depth_inc_cnt--;
        }

        for (auto sim_sim : sim_sims)
        {
            int eval_val = schneizel_sim_minimax_alphabeta_sync(sim_sim, false, depth - 1, depth_inc_cnt, depth_inc_max_move_cnt, alpha, beta);

            best_eval_val = eval_val < best_eval_val ? eval_val : best_eval_val;

            beta = eval_val < beta ? eval_val : beta;
            if (beta <= alpha)
            {
                break;
            }
        }

        return best_eval_val;
    }
}

void schneizel_sim_minimax_alphabeta_async(Simulation sim, bool white, int depth, int depth_inc_cnt, int depth_inc_max_move_cnt, int alpha, int beta, Evaluation *evals)
{
    int eval_val = schneizel_sim_minimax_alphabeta_sync(sim, white, depth, depth_inc_cnt, depth_inc_max_move_cnt, alpha, beta);
    evals[sim.idx] = Evaluation{eval_val, sim.move, sim.board};
}

std::vector<Evaluation> schneizel_minimax_alphabeta(Board *board, bool white, int depth, int depth_inc_cnt, int depth_inc_max_move_cnt)
{
    std::vector<Evaluation> best_moves;

    Evaluation evals[CHESS_BOARD_LEN];

    auto sims = board->simulate_all(white);

    int min = CHESS_EVAL_MIN_VAL;
    int max = CHESS_EVAL_MAX_VAL;

    int best_eval_val = white ? min : max;

    std::vector<std::thread> threads;

    for (auto sim : sims)
    {
        threads.push_back(std::thread(schneizel_sim_minimax_alphabeta_async, sim, white, depth, depth_inc_cnt, depth_inc_max_move_cnt, min, max, evals));
    }

    for (auto &th : threads)
    {
        th.join();
    }

    for (int i = 0; i < sims.size(); i++)
    {
        auto eval = evals[i];

        if ((white && eval.value > best_eval_val) || (!white && eval.value < best_eval_val))
        {
            best_moves.clear();

            best_eval_val = eval.value;

            best_moves.push_back(eval);
        }
        else if (eval.value == best_eval_val)
        {
            best_moves.push_back(eval);
        }
    }

    return best_moves;
}

void play(bool white, int depth)
{
    Board board;
    Move prev_move;

    OpeningEngine opening_engine("data/openings.data");
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
            do
            {
                std::string move_str;
                printf("Enter Move: ");
                std::cin >> move_str;
                prev_move = board.change(move_str, true);
            } while (!Move::is_valid(&prev_move));
        }
        else
        {
            if (move_cnt == 0)
            {
                // Default opening if white.
                prev_move = board.change("e4", true);
            }
            else
            {
                if (opening_stage)
                {
                    std::string move_str = opening_engine.next_move(&board, move_cnt);

                    if (move_str.empty())
                    {
                        printf("End of opening stage\n");
                        opening_stage = false;
                    }
                    else
                    {
                        prev_move = board.change(move_str, true);
                    }
                }

                if (!opening_stage)
                {
                    auto evals = board.minimax_alphabeta(true, depth, 7, 10);
                    int r = rand() % evals.size();
                    board.change(evals[r].move);
                    prev_move = evals[r].move;
                    printf("Ties: %d\n", evals.size());
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
            do
            {
                std::string move_str;
                printf("Enter Move: ");
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
                    printf("End of opening stage\n");
                    opening_stage = false;
                }
                else
                {
                    prev_move = board.change(move_str, false);
                }
            }

            if (!opening_stage)
            {
                auto evals = board.minimax_alphabeta(false, depth, 7, 10);
                int r = rand() % evals.size();
                board.change(evals[r].move);
                prev_move = evals[r].move;
                printf("Ties: %d\n", evals.size());
            }
        }

        move_cnt++;
    }
}

void selfplay_tiebreak(int depth, Model *model)
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

        // White:
        {
            if (move_cnt == 0)
            {
                // Default opening if white.
                prev_move = board.change("e4", true);
            }
            else
            {
                if (opening_stage)
                {
                    std::string move_str = opening_engine.next_move(&board, move_cnt);

                    if (move_str.empty())
                    {
                        printf("End of opening stage\n");
                        opening_stage = false;
                    }
                    else
                    {
                        prev_move = board.change(move_str, true);
                    }
                }

                if (!opening_stage)
                {
                    // auto evals = board.minimax_alphabeta(true, depth, 7, 10);
                    auto evals = schneizel_minimax_alphabeta(&board, true, depth, 7, 10);

                    int eval_idx = 0;
                    std::vector<int> a;

                    {
                        x->to_cpu();
                        one_hot_encode_chess_board_data(board.get_data(), x->data());
                        x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT)] = 1.0f;
                        x->data()[(CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 1)] = 0.0f;
                        auto p = model->forward(x);

                        for (int i = 0; i < evals.size(); i++)
                        {
                            float p_val = p->get_val(evals[i].move.src_square);
                            if (p_val >= 0.5f)
                            {
                                printf("Square: %d\tPiece: %c\tVal: %f\n", evals[i].move.src_square, board.get_piece(evals[i].move.src_square), p_val);
                                eval_idx = i;
                                a.push_back(i);
                            }
                        }

                        p->reshape(Shape(1, CHESS_ROW_CNT, CHESS_COL_CNT));
                        p->print();
                        delete p;
                    }

                    board.change(evals[eval_idx].move);
                    prev_move = evals[eval_idx].move;
                    printf("Ties: %d\n", evals.size());
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

        // Black:
        {
            if (opening_stage)
            {
                std::string move_str = opening_engine.next_move(&board, move_cnt);

                if (move_str.empty())
                {
                    printf("End of opening stage\n");
                    opening_stage = false;
                }
                else
                {
                    prev_move = board.change(move_str, false);
                }
            }

            if (!opening_stage)
            {
                auto evals = board.minimax_alphabeta(false, depth, 7, 10);
                int r = rand() % evals.size();
                board.change(evals[r].move);
                prev_move = evals[r].move;
                printf("Ties: %d\n", evals.size());
            }
        }

        move_cnt++;
    }

    delete x;
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

void train(Model *model, int epochs, int batch_size)
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

                auto p = model->forward(x);

                if (batch_idx % 100 == 0)
                {
                    float loss = model->loss(p, oh_y);
                    float acc = model->accuracy(p, oh_y, chess_classification_accuracy_fn);
                    fprintf(train_csv, "%d,%d,%f,%f\n", epoch, batch_idx, loss, acc);
                }

                model->backward(p, oh_y);
                model->step();

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

void compare_models(int epochs, int batch_size)
{
    Shape x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape y_shape(batch_size, CHESS_BOARD_LEN);

    {
        auto model = new Model(new Xavier());

        model->linear(x_shape, 1024, new ReLU());
        model->linear(1024, new ReLU());
        model->linear(512, new ReLU());
        model->linear(512, new ReLU());
        model->linear(128, new ReLU());
        model->linear(y_shape, new Sigmoid());

        model->set_loss(new CrossEntropy());
        model->set_optimizer(new SGDMomentum(model->parameters(), 0.01f, ZERO_NN_BETA_1));

        model->summarize();

        train(model, epochs, batch_size);

        model->save_parameters("temp/model.nn");

        delete model;
    }
}

void cont_training(int epochs, int batch_size)
{
    Shape x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape y_shape(batch_size, CHESS_BOARD_LEN);

    {
        auto model = new Model(new Xavier());

        model->linear(x_shape, 1024, new ReLU());
        model->linear(1024, new ReLU());
        model->linear(512, new ReLU());
        model->linear(512, new ReLU());
        model->linear(128, new ReLU());
        model->linear(y_shape, new Sigmoid());

        model->set_loss(new CrossEntropy());
        model->set_optimizer(new SGDMomentum(model->parameters(), 0.001f, ZERO_NN_BETA_1));

        model->load_parameters("temp/model.nn");

        model->summarize();

        train(model, epochs, batch_size);

        // model->save_parameters("temp/model.nn");

        delete model;
    }
}

int main()
{
    srand(time(NULL));

    // export_pgn("data/all.pgn");

    // compare_models(10, 64);

    // cont_training(10, 64);

    Shape x_shape(1, CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT + 2);
    Shape y_shape(1, CHESS_BOARD_LEN);

    auto model = new Model(new Xavier());
    {
        model->linear(x_shape, 1024, new ReLU());
        model->linear(1024, new ReLU());
        model->linear(512, new ReLU());
        model->linear(512, new ReLU());
        model->linear(128, new ReLU());
        model->linear(y_shape, new Sigmoid());

        model->set_loss(new CrossEntropy());
        model->set_optimizer(new SGDMomentum(model->parameters(), 0.001f, ZERO_NN_BETA_1));

        model->load_parameters("temp/model.nn");
    }

    selfplay_tiebreak(3, model);

    delete model;

    return 0;
}