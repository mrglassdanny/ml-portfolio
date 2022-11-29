#include <stdio.h>

#include "chess.h"

void self_play(int depth)
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

		if (board.is_checkmate(false))
		{
			printf("\nWHITE CHECKMATED!\n");
			break;
		}
		else if (!board.has_moves(true))
		{
			printf("\nWHITE STALEMATED!\n");
			break;
		}

		prev_move = board.change_minimax_async(true, 5);

		move_cnt++;

		printf("\nBLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
		board.print(prev_move);

		if (board.is_checkmate(true))
		{
			printf("\BLACK CHECKMATED!\n");
			break;
		}
		else if (!board.has_moves(false))
		{
			printf("\BLACK STALEMATED!\n");
			break;
		}

		prev_move = board.change_minimax_async(false, 2);

		move_cnt++;
	}
}

void play(bool white, int depth)
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

		if (white)
		{
			std::string move_str;
			printf("Move: ");
			std::cin >> move_str;
			prev_move = board.change(move_str, true);
		}
		else
		{
			prev_move = board.change_minimax_async(true, depth);
		}

		move_cnt++;

		printf("\nBLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
		board.print(prev_move);

		if (!white)
		{
			std::string move_str;
			printf("Move: ");
			std::cin >> move_str;
			prev_move = board.change(move_str, false);
		}
		else
		{
			prev_move = board.change_minimax_async(false, depth);
		}

		move_cnt++;
	}
}

int main()
{
	srand(time(NULL));

	self_play(6);

	return 0;
}