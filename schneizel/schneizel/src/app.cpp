#include <stdio.h>

#include "chess.h"

int main()
{

	Board board;

	board.print(false);

	auto vec = board.get_sims(true);

	for (Board b : vec)
	{
		b.print(false);
	}

	return 0;
}