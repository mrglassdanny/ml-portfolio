#include <stdio.h>

#include <ATen/ATen.h>

#include "chess.h"

using namespace chess;

int main()
{
	srand(time(NULL));

	bool white = true;

	auto board = Openings::create_rand_d4(&white);
	board.print_analysis(BoardAnalysisType::PieceTypes);

	while (!board.game_over())
	{
		board.change_minimax(white, 2);
		white = !white;
		board.print_analysis(BoardAnalysisType::PieceTypes);
	}

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}