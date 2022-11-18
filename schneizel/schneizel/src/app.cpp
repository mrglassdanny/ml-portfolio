#include <stdio.h>

//#include <ATen/ATen.h>

#include "fastchess.h"

using namespace fastchess;

int main()
{
	srand(time(NULL));

	Board board;

	auto ms = board.get_all_moves(true);
	for (auto m : ms)
	{
		board.change(m);
		board.print();
		board.reset();
	}

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}