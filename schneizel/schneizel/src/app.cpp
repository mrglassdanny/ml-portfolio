#include <stdio.h>

#include "chess.h"
#include "fastchess.h"

int main()
{
	srand(time(NULL));

	chess::Board c_board;
	fastchess::Board fc_board;

	// c_board.change_minimax(true, 7);
	// c_board.pretty_print();

	fc_board.change_minimax(true, 5);
	fc_board.print();

	return 0;
}