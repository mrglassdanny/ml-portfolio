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
		auto move = board.change();
		printf("=================================\n");
		board.print_status();
		printf("Move: %s\n", board.convert_move_to_an_move(move).c_str());
		board.print();
	}

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}