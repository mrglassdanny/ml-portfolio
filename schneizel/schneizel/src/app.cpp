#include <stdio.h>

#include "chess.h"

int main()
{

	Board board;

	board.change("d4");
	board.print(false);
	board.change("d5");
	board.print(false);
	board.change(board.get_random_move(&board));
	board.print(false);
	board.change(board.get_random_move(&board));
	board.print(false);
	board.change(board.get_random_move(&board));
	board.print(false);
	board.change(board.get_random_move(&board));
	board.print(false);
	board.change(board.get_random_move(&board));
	board.print(false);

	auto vec = board.simulate_all_legal_moves();
	for (Board b : vec)
	{
		// b.print(false);
	}

	return 0;
}