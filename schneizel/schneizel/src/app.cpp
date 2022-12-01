#include <stdio.h>

#include "chess.h"

void self_play(int white_depth, int black_depth)
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

		if (board.check_state_.white_checked)
		{
			printf("======================================================== WHITE IN CHECK!\n");
		}

		prev_move = board.change_minimax_async(true, white_depth);

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

		if (board.check_state_.black_checked)
		{
			printf("======================================================== BLACK IN CHECK!\n");
		}

		prev_move = board.change_minimax_async(false, black_depth);

		move_cnt++;
	}
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

	self_play(5, 5);
	// play(true, 1);

	return 0;
}