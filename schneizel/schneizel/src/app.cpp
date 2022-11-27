#include <stdio.h>

#include "chess.h"
#include "fastchess.h"

int main()
{
	srand(time(NULL));

	fastchess::Board fc_board;

	for (int i = 0; i < 4; i++)
	{
		fc_board.change_minimax_async(true, 4);
		fc_board.print();

		fc_board.change_minimax_async(false, 4);
		fc_board.print();
	}

	fc_board.reset();

	for (int i = 0; i < 4; i++)
	{
		fc_board.change_minimax_sync(true, 4);
		fc_board.print();

		fc_board.change_minimax_sync(false, 4);
		fc_board.print();
	}

	return 0;
}