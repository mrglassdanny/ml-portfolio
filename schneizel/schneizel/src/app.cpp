#include <stdio.h>

#include "chess.h"

int main()
{

	Board board;

	board.print(false);
	board.change("e4", true);
	board.print(false);
	board.change("e5", false);
	board.print(false);
	board.change(board.get_random_move(true, &board), true);
	board.print(false);

	board.print_influence();
	board.print_float();

	auto vec = board.get_sims(false);

	for (Board b : vec)
	{
		b.print(false);
	}

	return 0;
}