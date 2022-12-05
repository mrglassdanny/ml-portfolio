#include <stdio.h>

#include <zero/mod.cuh>

#include "chess.cuh"

using namespace zero::core;
using namespace zero::nn;
using namespace chess;

struct Game
{
    std::vector<Board> boards;
    int lbl;
};

Game self_play(int white_depth, int black_depth, bool print, Model *model)
{
    Board board;
    Move prev_move;

    Game game;
    game.lbl = 0;

    int move_cnt = 0;

    while (move_cnt < 100)
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

            game.lbl = 1;
            break;
        }
        else if (!board.has_moves(true))
        {
            if (print)
                printf("WHITE STALEMATED!\n");

            game.lbl = 0;
            break;
        }

        if (board.is_check(false, false) && print)
        {
            printf("======================================================== WHITE IN CHECK!\n");
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

            game.lbl = -1;
            break;
        }
        else if (!board.has_moves(false))
        {
            if (print)
                printf("BLACK STALEMATED!\n");

            game.lbl = 0;
            break;
        }

        if (board.is_check(true, false) && print)
        {
            printf("======================================================== BLACK IN CHECK!\n");
        }

        prev_move = board.change_minimax_async(false, black_depth);
        Board cpy_board2;
        cpy_board2.copy(&board);
        game.boards.push_back(cpy_board2);

        move_cnt++;
    }

    return game;
}

int main()
{
    srand(time(NULL));

    auto x = Tensor::zeros(false, Shape(1, 6, 8, 8));
    auto y = Tensor::zeros(true, Shape(1, 1));

    auto model = new Model();
    model->hadamard_product(x->shape(), 16, layer::ActivationType::Tanh);
    model->matrix_product(16, layer::ActivationType::Tanh);
    model->linear(y->shape(), layer::ActivationType::Tanh);

    model->set_loss(new loss::MSE());
    model->set_optimizer(new optim::SGD(model->parameters(), 0.1f));

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

                auto p = model->forward(x);
                printf("LOSS: %f\n", model->loss(p, y));
                model->backward(p, y);
                model->step();

                delete p;
            }
        }

        printf("GAME COUNT: %d\tLABEL: %d\n", game_cnt, game.lbl);
    }

    delete x;
    delete y;

    return 0;
}