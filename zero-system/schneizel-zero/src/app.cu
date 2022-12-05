#include <stdio.h>

#include <zero/mod.cuh>

#include "chess.cuh"

using namespace zero::core;
using namespace zero::nn;
using namespace chess;

struct Game
{
    std::vector<chess::Board> boards;
    int lbl;
};

Game self_play(int white_depth, int black_depth, Model *model)
{
    chess::Board board;
    chess::Move prev_move;

    Game game;

    std::vector<Model *> models;
    models.push_back(model);
    for (int i = 0; i < 49; i++)
    {
        auto cpy_model = new Model(true);
        cpy_model->hadamard_product(model->input_shape(), 16, layer::ActivationType::Tanh);
        cpy_model->matrix_product(16, layer::ActivationType::Tanh);
        cpy_model->linear(model->output_shape(), layer::ActivationType::Tanh);

        cpy_model->share_parameters(model->parameters());

        cpy_model->set_loss(new loss::MSE());

        models.push_back(cpy_model);
    }

    int move_cnt = 0;

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

        if (board.check_state_.white_checked)
        {
            printf("======================================================== WHITE IN CHECK!\n");
        }

        prev_move = board.change_minimax_async(true, white_depth, models);
        chess::Board cpy_board;
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

        if (board.check_state_.black_checked)
        {
            printf("======================================================== BLACK IN CHECK!\n");
        }

        prev_move = board.change_minimax_async(false, black_depth, models);
        chess::Board cpy_board2;
        cpy_board2.copy(&board);
        game.boards.push_back(cpy_board2);

        move_cnt++;
    }

    return game;
}

// void play(bool play_as_white, int cpu_depth)
// {
//     chess::Board board;
//     chess::Move prev_move;

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

    chess::Board board;

    Tensor *x = Tensor::zeros(false, Shape(1, 6, 8, 8));
    board.one_hot_encode(x->data());
    x->to_cuda();
    auto y = Tensor::zeros(true, Shape(1, 1));

    auto model = new Model();
    model->hadamard_product(x->shape(), 16, layer::ActivationType::Tanh);
    model->matrix_product(16, layer::ActivationType::Tanh);
    model->linear(y->shape(), layer::ActivationType::Tanh);

    model->set_loss(new loss::MSE());
    model->set_optimizer(new optim::SGD(model->parameters(), 0.01f));

    model->summarize();

    self_play(1, 1, model);

    return 0;
}