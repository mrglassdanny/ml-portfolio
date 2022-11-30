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

		prev_move = board.change_minimax_async(true, white_depth);

		move_cnt++;

		printf("\nBLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
		board.print(prev_move);

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

		if (play_as_white)
		{
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

		if (!play_as_white)
		{
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

	self_play(4, 3);

	return 0;
}