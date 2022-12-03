#include <stdio.h>

#include <zero/mod.cuh>

#include "chess.cuh"

struct Game
{
    zero::core::Tensor *one_hot_board_data;
    int move_cnt;
    int lbl;
};

Game self_play(int white_depth, int black_depth)
{
    chess::Board board;
    chess::Move prev_move;

    Game game;
    std::vector<chess::Board> boards;

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

        prev_move = board.change_minimax_async(true, white_depth);
        chess::Board cpy_board;
        cpy_board.copy(&board);
        boards.push_back(cpy_board);

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

        prev_move = board.change_minimax_async(false, black_depth);
        chess::Board cpy_board2;
        cpy_board2.copy(&board);
        boards.push_back(cpy_board2);

        move_cnt++;
    }

    game.move_cnt = move_cnt;

    game.one_hot_board_data = zero::core::Tensor::zeros(false, zero::core::Shape(game.move_cnt, 8, 8, 6));
    for (auto b : boards)
    {
        b.one_hot_encode(&game.one_hot_board_data->data()[(8 * 8 * 6) * move_cnt]);
    }

    game.one_hot_board_data->print();

    return game;
}

void play(bool play_as_white, int cpu_depth)
{
    chess::Board board;
    chess::Move prev_move;

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
            break;
        }
        else if (!board.has_moves(true))
        {
            printf("WHITE STALEMATED!\n");
            break;
        }

        if (play_as_white)
        {
            auto moves = board.get_all_moves(true);
            for (auto move : moves)
            {
                printf("Move: %s\n", board.convert_move_to_move_str(move).c_str());
            }

            std::string move_str;
            printf("Move: ");
            std::cin >> move_str;
            prev_move = board.change(move_str, true);
        }
        else
        {
            prev_move = board.change_minimax_async(true, cpu_depth);
        }

        move_cnt++;

        printf("\nBLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
        board.print(prev_move);

        if (board.is_checkmate(true, true))
        {
            printf("BLACK CHECKMATED!\n");
            break;
        }
        else if (!board.has_moves(false))
        {
            printf("BLACK STALEMATED!\n");
            break;
        }

        if (!play_as_white)
        {
            auto moves = board.get_all_moves(false);
            for (auto move : moves)
            {
                printf("Move: %s\n", board.convert_move_to_move_str(move).c_str());
            }

            std::string move_str;
            printf("Move: ");
            std::cin >> move_str;
            prev_move = board.change(move_str, false);
        }
        else
        {
            prev_move = board.change_minimax_async(false, cpu_depth);
        }

        move_cnt++;
    }
}

int main()
{
    srand(time(NULL));

    Game game = self_play(3, 3);

    return 0;
}