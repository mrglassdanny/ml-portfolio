#include <stdio.h>

#include "chess.h"

int main()
{

	Board board;

	board.print(false);

	auto vec = board.simulate_all_legal_moves(true);

	for (Board b : vec)
	{
		b.print(false);
	}

	return 0;
}