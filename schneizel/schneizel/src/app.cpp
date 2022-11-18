#include <stdio.h>

#include <ATen/ATen.h>

#include "fastchess.h"

using namespace fastchess;

int main()
{
	srand(time(NULL));

	Board board;

	board.print();

	board.change(Move{11, 19});
	board.print();

	for (auto move : board.get_moves(2))
	{
		board.change(move);
		board.print();
		board.reset();
	}

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}