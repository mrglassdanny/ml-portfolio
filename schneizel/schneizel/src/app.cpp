#include <stdio.h>

#include "chess.h"
#include "fastchess.h"

int main()
{
	srand(time(NULL));

	chess::Board c_board;
	fastchess::Board fc_board;

	// c_board.change_minimax(true, 6);
	// c_board.pretty_print();

	fc_board.change_minimax_async(true, 6);
	fc_board.print();

	// for (int i = 0; i < 100; i++)
	// {
	// 	fc_board.change_minimax(true, 5);
	// 	fc_board.print();

	// 	fc_board.change_random(false);
	// 	fc_board.print();
	// }

	return 0;
}