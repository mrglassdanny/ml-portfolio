#include <stdio.h>

#include "chess.h"
#include "fastchess.h"

void self_play(int depth)
{
	fastchess::Board board;
	fastchess::Move prev_move;

	int move_cnt = 0;

	while (true)
	{
		printf("\n\nWHITE TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());

		if (move_cnt == 0)
		{
			board.print();
		}
		else
		{
			board.print(prev_move);
		}

		prev_move = board.change_minimax_async(true, depth);

		move_cnt++;

		printf("\n\BLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
		board.print(prev_move);

		prev_move = board.change_minimax_async(false, depth);

		move_cnt++;
	}
}

void play(bool white, int depth)
{
	fastchess::Board board;
	fastchess::Move prev_move;

	int move_cnt = 0;

	while (true)
	{
		printf("\n\nWHITE TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());

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
			auto moves = board.get_all_moves(true);
			for (auto move : moves)
			{
				printf("MOVE: %s (%d->%d)\n", board.convert_move_to_algnote_move(move).c_str(),
					   move.src_square, move.dst_square);
			}

			fastchess::Move move;
			printf("Source square: ");
			std::cin >> move.src_square;
			printf("Destination square: ");
			std::cin >> move.dst_square;
			board.change(move);

			prev_move = move;
		}
		else
		{
			prev_move = board.change_minimax_async(true, depth);
		}

		move_cnt++;

		printf("\n\BLACK TURN\tCURRENT MATERIAL EVAL: %d\n", board.evaluate_material());
		board.print(prev_move);

		if (!white)
		{
			auto moves = board.get_all_moves(false);
			for (auto move : moves)
			{
				printf("MOVE: %s (%d->%d)\n", board.convert_move_to_algnote_move(move).c_str(),
					   move.src_square, move.dst_square);
			}

			fastchess::Move move;
			printf("Source square: ");
			std::cin >> move.src_square;
			printf("Destination square: ");
			std::cin >> move.dst_square;
			board.change(move);

			prev_move = move;
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

	play(true, 4);

	return 0;
}