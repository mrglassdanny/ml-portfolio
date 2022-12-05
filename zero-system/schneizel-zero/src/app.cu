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

Game self_play(int white_depth, int black_depth, Model *model)
{
    Board board;
    Move prev_move;

    Game game;

    int move_cnt = 0;

    auto material_evaluator = new MaterialEvaluator();
    auto model_evaluator = new ModelEvaluator(model->copy());

    std::vector<Evaluator *> model_evaluators;
    model_evaluators.push_back(model_evaluator);
    for (int i = 0; i < 74; i++)
    {
        model_evaluators.push_back(new ModelEvaluator(model->copy()));
    }

    while (true)
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

        if (board.is_checkmate(false, true))
        {
            printf("WHITE CHECKMATED!\n");
            game.lbl = 1;
            break;
        }
        else if (!board.has_moves(true))
        {
            printf("WHITE STALEMATED!\n");
            game.lbl = 0;
            break;
        }

        if (board.is_check(false, false))
        {
            printf("======================================================== WHITE IN CHECK!\n");
        }

        // prev_move = board.change_minimax_sync(true, white_depth, model_evaluator);
        prev_move = board.change_minimax_async(true, white_depth, model_evaluators);
        Board cpy_board;
        cpy_board.copy(&board);
        game.boards.push_back(cpy_board);

        move_cnt++;

        printf("\nBLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
        board.print(prev_move);

        if (board.is_checkmate(true, true))
        {
            printf("BLACK CHECKMATED!\n");
            game.lbl = -1;
            break;
        }
        else if (!board.has_moves(false))
        {
            printf("BLACK STALEMATED!\n");
            game.lbl = 0;
            break;
        }

        if (board.is_check(true, false))
        {
            printf("======================================================== BLACK IN CHECK!\n");
        }

        prev_move = board.change_minimax_async(false, black_depth, material_evaluator);
        Board cpy_board2;
        cpy_board2.copy(&board);
        game.boards.push_back(cpy_board2);

        move_cnt++;
    }

    delete material_evaluator;
    delete model_evaluator;

    for (int i = 0; i < 75; i++)
    {
        delete model_evaluators[i];
    }

    return game;
}

int main()
{
    srand(time(NULL));

    auto x = Tensor::zeros(false, Shape(1, 6, 8, 8));
    auto y = Tensor::zeros(true, Shape(1, 1));

    auto model = new Model();
    model->hadamard_product(x->shape(), 4, layer::ActivationType::Tanh);
    model->matrix_product(4, layer::ActivationType::Tanh);
    model->linear(y->shape(), layer::ActivationType::Tanh);

    model->set_loss(new loss::MSE());
    model->set_optimizer(new optim::SGD(model->parameters(), 0.1f));

    model->summarize();

    while (true)
    {
        auto game = self_play(1, 3, model);
        for (auto board : game.boards)
        {
            x->to_cpu();
            board.one_hot_encode(x->data());
            x->to_cuda();

            y->set_val(0, game.lbl);

            auto p = model->forward(x);
            model->backward(p, y);
            model->step();

            delete p;
        }
    }

    delete x;
    delete y;

    return 0;
}