#include <stdio.h>

#include <ATen/ATen.h>

#include "chess.h"

using namespace chess;

int main()
{
	srand(time(NULL));

	Board board;

	while (!board.game_over())
	{
		printf("\n================================= =================================\n");
		board.print_status();
		board.change();
		board.print();
		board.print_status();
		printf("================================= =================================\n");
	}

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}