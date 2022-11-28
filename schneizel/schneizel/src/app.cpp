#include <stdio.h>

#include "chess.h"
#include "fastchess.h"

int main()
{
	srand(time(NULL));

	fastchess::Board fc_board;

	for (int i = 0; i < 50; i++)
	{
		printf("CURRENT EVAL: %d WHITE TURN\n", fc_board.evaluate_material());
		fc_board.change_minimax_async(true, 4);
		fc_board.print();

		printf("CURRENT EVAL: %d BLACK TURN\n", fc_board.evaluate_material());
		fc_board.change_minimax_async(false, 4);
		fc_board.print();
	}

	return 0;
}