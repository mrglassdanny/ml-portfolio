#include <stdio.h>
#include <conio.h>

#include <zero/mod.cuh>

#include "chess.cuh"

using namespace zero::core;
using namespace zero::nn;

using namespace chess;

void export_pgn(const char *path)
{
    auto pgn_games = PGN::import(path, FileUtils::get_file_size(path));

    FILE *train_data_file = fopen("temp/train.data", "wb");
    FILE *train_lbl_file = fopen("temp/train.lbl", "wb");

    char data_buf[CHESS_BOARD_LEN];
    int lbl_buf;

    int game_cnt = 0;

    long move_cnt = 0;

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
            if (p_val >= 0.33f)
            {
                correct_cnt++;
            }
        }
        else if (y_val < 0.0f)
        {
            // Black win.
            if (p_val <= -0.33f)
            {
                correct_cnt++;
            }
        }
        else
        {
            // Draw.
            if (p_val < 0.33f && p_val > -0.33f)
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

    // Small:
    model->linear(x_shape, 1024, new Tanh());
    model->linear(1024, new Tanh());
    model->linear(512, new Tanh());
    model->linear(128, new Tanh());
    model->linear(32, new Tanh());
    model->linear(y_shape, new Tanh());

    model->set_loss(new MSE());
    model->set_optimizer(new SGDMomentum(model->parameters(), 0.1f, ZERO_NN_BETA_1));

    model->summarize();

    if (params_path != nullptr)
    {
        model->load_parameters(params_path);
    }

    return model;
}

Model *get_model_test(int batch_size, const char *params_path)
{
    Shape x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT);
    Shape y_shape(batch_size, 1);

    auto model = new Model(new Xavier());

    // Small:
    model->hadamard_product(x_shape, 32, new Tanh());
    model->matrix_product(32, new Tanh());
    model->linear(512, new Tanh());
    model->linear(128, new Tanh());
    model->linear(32, new Tanh());
    model->linear(y_shape, new Tanh());

    model->set_loss(new MSE());
    model->set_optimizer(new SGDMomentum(model->parameters(), 0.1f, ZERO_NN_BETA_1));

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

    auto model = get_model_test(batch_size, nullptr);

    auto sw = new CudaStopWatch();

    {
        const char *data_path = "temp/train.data";
        const char *lbl_path = "temp/train.lbl";

        int data_size = CHESS_BOARD_LEN;
        int x_size = (CHESS_BOARD_CHANNEL_CNT * CHESS_ROW_CNT * CHESS_COL_CNT);

        long long data_file_size = FileUtils::get_file_size(data_path);
        // size_t data_cnt = data_file_size / data_size;
        size_t data_cnt = 1000000;

        int batch_cnt = data_cnt / batch_size;

        FILE *data_file = fopen(data_path, "rb");
        FILE *lbl_file = fopen(lbl_path, "rb");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, data_cnt - 1);

        {
            FILE *train_csv = fopen("temp/train.csv", "w");
            fprintf(train_csv, "epoch,batch,loss,accuracy,elapsed_secs\n");

            bool quit = false;

            auto x = Tensor::zeros(false, x_shape);
            auto y = Tensor::zeros(false, y_shape);

            char data_buf[CHESS_BOARD_LEN];
            int lbl_buf;

            sw->start();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // if (epoch == 5)
                // {
                //     model->save_parameters("temp/model-5.nn");
                // }
                // else if (epoch == 8)
                // {
                //     model->save_parameters("temp/model-8.nn");
                // }
                // else if (epoch == 10)
                // {
                //     model->save_parameters("temp/model-10.nn");
                // }
                // else if (epoch == 15)
                // {
                //     model->save_parameters("temp/model-15.nn");
                // }

                for (int batch_idx = 0; batch_idx < batch_cnt; batch_idx++)
                {
                    x->zeros();
                    y->zeros();

                    x->to_cpu();
                    y->to_cpu();

                    for (int i = 0; i < batch_size; i++)
                    {
                        long offset = dist(gen);
                        fseek(data_file, offset * data_size, SEEK_SET);
                        fseek(lbl_file, offset * sizeof(int), SEEK_SET);

                        fread(data_buf, data_size, 1, data_file);
                        fread(&lbl_buf, sizeof(int), 1, lbl_file);

                        Board::one_hot_encode_chess_board_data(data_buf, &x->data()[i * x_size]);
                        y->data()[i] = (float)lbl_buf;
                    }

                    auto p = model->forward(x);

                    if (batch_idx % 100 == 0)
                    {
                        float loss = model->loss(p, y);
                        float acc = model->accuracy(p, y, chess_accuracy_fn);

                        sw->stop();
                        fprintf(train_csv, "%d,%d,%f,%f,%f\n", epoch, batch_idx, loss, acc, sw->get_elapsed_seconds());
                        sw->start();
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

    // model->save_parameters("temp/model.nn");

    sw->stop();
    delete sw;

    delete model;
}

void test(int batch_size)
{
    Shape x_shape(batch_size, CHESS_BOARD_CHANNEL_CNT, CHESS_ROW_CNT, CHESS_COL_CNT);
    Shape y_shape(batch_size, 1);

    auto model = get_model(batch_size, "data/small-model-20.nn");

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
            FILE *test_csv = fopen("temp/test.csv", "w");
            fprintf(test_csv, "batch,acc\n");

            auto x = Tensor::zeros(false, x_shape);
            auto y = Tensor::zeros(false, y_shape);

            char data_buf[CHESS_BOARD_LEN];
            int lbl_buf;

            for (int batch_idx = 0; batch_idx < batch_cnt; batch_idx++)
            {
                x->zeros();
                y->zeros();

                x->to_cpu();
                y->to_cpu();

                for (int i = 0; i < batch_size; i++)
                {
                    fread(data_buf, data_size, 1, data_file);
                    fread(&lbl_buf, sizeof(int), 1, lbl_file);

                    Board::one_hot_encode_chess_board_data(data_buf, &x->data()[i * x_size]);
                    y->data()[i] = (float)lbl_buf;
                }

                auto p = model->forward(x);

                float acc = model->accuracy(p, y, chess_accuracy_fn);
                fprintf(test_csv, "%d,%f\n", batch_idx, acc);

                delete p;

                if (_kbhit())
                {
                    if (_getch() == 'q')
                    {
                        break;
                    }
                }
            }

            delete x;
            delete y;

            fclose(test_csv);
        }

        fclose(data_file);
        fclose(lbl_file);
    }

    delete model;
}

void play(bool white, int depth, Model *model)
{
    Board board;
    Move prev_move;

    OpeningEngine opening_engine("data/openings.data");
    bool opening_stage = true;

    std::vector<Model *> models;
    for (int i = 0; i < 64; i++)
    {
        models.push_back(model->copy());
    }

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
                    auto eval_dataset = board.minimax_alphabeta(true, depth, 5, 12, model);

                    int max_eval_idx = 0;

                    for (int eval_data_idx = 0; eval_data_idx < eval_dataset.size(); eval_data_idx++)
                    {
                        auto move = eval_dataset[eval_data_idx].move;
                        printf("Src: %d\tDst: %d\tPiece: %c\tEvaluation: %f\tDepth: %d\n", move.src_square, move.dst_square, board.get_piece(move.src_square), eval_dataset[eval_data_idx].eval.value, eval_dataset[eval_data_idx].eval.depth);
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
                auto eval_dataset = board.minimax_alphabeta(false, depth, 5, 12, model);

                int max_eval_idx = 0;

                for (int eval_data_idx = 0; eval_data_idx < eval_dataset.size(); eval_data_idx++)
                {
                    auto move = eval_dataset[eval_data_idx].move;
                    printf("Src: %d\tDst: %d\tPiece: %c\tEvaluation: %f\tDepth: %d\n", move.src_square, move.dst_square, board.get_piece(move.src_square), eval_dataset[eval_data_idx].eval.value, eval_dataset[eval_data_idx].eval.depth);
                }

                board.change(eval_dataset[max_eval_idx].move);
                prev_move = eval_dataset[max_eval_idx].move;
                printf("TIES: %d\n", eval_dataset.size());
            }
        }

        move_cnt++;
    }

    for (int i = 0; i < 64; i++)
    {
        delete models[i];
    }
}

int main()
{
    srand(time(NULL));

    auto model = get_model(1, "data/small-model.nn");
    play(true, 4, model);
    delete model;

    return 0;
}