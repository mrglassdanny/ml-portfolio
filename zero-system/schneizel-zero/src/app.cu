#include <stdio.h>

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

int main()
{
    srand(time(NULL));

    auto pgn_games = PGN::import("data/data.pgn");

    int i = 0;
    for (auto pgn_game : pgn_games)
    {
        Board board;
        bool white = true;

        for (auto move_str : pgn_game->move_strs)
        {
            if (i == -1)
            {
                printf("Move: %s\n", move_str.c_str());
                auto move = board.change(move_str, white);
                white = !white;
                board.print(move);
            }
            else
            {
                auto move = board.change(move_str, white);
                white = !white;
            }
        }
        printf("Game: %d\n", i++);
        // system("cls");
    }

    return 0;
}