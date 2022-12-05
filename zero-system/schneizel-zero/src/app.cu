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
    auto model_evaluator = new ModelEvaluator(model);

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

        prev_move = board.change_minimax_sync(true, white_depth, model_evaluator);
        printf("MODEL SYNC EVAL CNT: %d\t", model_evaluator->get_count());
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

        // prev_move = board.change_minimax_async(false, black_depth, material_evaluator);
        prev_move = board.change_minimax_sync(false, black_depth, material_evaluator);
        printf("MATERIAL SYNC EVAL CNT: %d\n", material_evaluator->get_count());
        Board cpy_board2;
        cpy_board2.copy(&board);
        game.boards.push_back(cpy_board2);

        move_cnt++;
    }

    delete material_evaluator;
    delete model_evaluator;

    return game;
}

// void play(bool play_as_white, int cpu_depth)
// {
//     Board board;
//     Move prev_move;

//     int move_cnt = 0;

//     while (true)
//     {
//         printf("\nWHITE TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());

//         if (move_cnt == 0)
//         {
//             board.print();
//         }
//         else
//         {
//             board.print(prev_move);
//         }

//         if (board.is_checkmate(false, true))
//         {
//             printf("WHITE CHECKMATED!\n");
//             break;
//         }
//         else if (!board.has_moves(true))
//         {
//             printf("WHITE STALEMATED!\n");
//             break;
//         }

//         if (play_as_white)
//         {
//             auto moves = board.get_all_moves(true);
//             for (auto move : moves)
//             {
//                 printf("Move: %s\n", board.convert_move_to_move_str(move).c_str());
//             }

//             std::string move_str;
//             printf("Move: ");
//             std::cin >> move_str;
//             prev_move = board.change(move_str, true);
//         }
//         else
//         {
//             prev_move = board.change_minimax_async(true, cpu_depth);
//         }

//         move_cnt++;

//         printf("\nBLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
//         board.print(prev_move);

//         if (board.is_checkmate(true, true))
//         {
//             printf("BLACK CHECKMATED!\n");
//             break;
//         }
//         else if (!board.has_moves(false))
//         {
//             printf("BLACK STALEMATED!\n");
//             break;
//         }

//         if (!play_as_white)
//         {
//             auto moves = board.get_all_moves(false);
//             for (auto move : moves)
//             {
//                 printf("Move: %s\n", board.convert_move_to_move_str(move).c_str());
//             }

//             std::string move_str;
//             printf("Move: ");
//             std::cin >> move_str;
//             prev_move = board.change(move_str, false);
//         }
//         else
//         {
//             prev_move = board.change_minimax_async(false, cpu_depth);
//         }

//         move_cnt++;
//     }
// }

int main()
{
    srand(time(NULL));

    Board board;

    Tensor *x = Tensor::zeros(false, Shape(1, 6, 8, 8));
    board.one_hot_encode(x->data());
    x->to_cuda();
    auto y = Tensor::zeros(true, Shape(1, 1));

    auto model = new Model();
    model->hadamard_product(x->shape(), 16, layer::ActivationType::Tanh);
    model->matrix_product(16, layer::ActivationType::Tanh);
    model->linear(y->shape(), layer::ActivationType::Tanh);

    model->set_loss(new loss::MSE());
    model->set_optimizer(new optim::SGD(model->parameters(), 0.1f));

    model->summarize();

    self_play(1, 1, model);

    return 0;
}