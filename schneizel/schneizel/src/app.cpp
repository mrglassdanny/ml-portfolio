#include <stdio.h>

#include "chess.h"

int main()
{
	int *board = init_board();
	print_board(board);
	return 0;
}