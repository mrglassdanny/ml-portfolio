#include <stdio.h>

#include <ATen/ATen.h>

#include "chess.h"

using namespace chess;

int main()
{
	srand(time(NULL));

	bool white = true;

	auto board = Openings::create(OpeningType::GrunfeldDefense);
	board.print_analysis(BoardAnalysisType::PieceTypes);

	board.change("Qa4", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes);

	board.change("b5", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes);

	board.change("Qxb5", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes);

	board.change("c6", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes);

	board.change("Qb7", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes);

	board.change("Na6", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes);

	board.change("Qxc6", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes);

	{
		auto sims = board.simulate_all_moves(white);
		for (auto sim : sims)
		{
			printf("%f\n", sim.minimax(3, white, -1000.f, 1000.f));
			sim.print_analysis(BoardAnalysisType::PieceTypes);
		}
	}

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}