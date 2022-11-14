#include <stdio.h>

#include <ATen/ATen.h>

#include "chess.h"

using namespace chess;

int main()
{
	srand(time(NULL));

	bool white = true;

	auto board = Openings::create(OpeningType::GrunfeldDefense);
	board.print_analysis(BoardAnalysisType::PieceTypes, white);
	board.print_analysis(BoardAnalysisType::AttackOpportunities, white);

	board.change("Qa4", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes, white);
	board.print_analysis(BoardAnalysisType::AttackOpportunities, white);

	board.change("b5", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes, white);
	board.print_analysis(BoardAnalysisType::AttackOpportunities, white);

	board.change("Qxb5", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes, white);
	board.print_analysis(BoardAnalysisType::AttackOpportunities, white);

	board.change("c6", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes, white);
	board.print_analysis(BoardAnalysisType::AttackOpportunities, white);

	board.change("Qb7", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes, white);
	board.print_analysis(BoardAnalysisType::AttackOpportunities, white);

	board.change("Na6", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes, white);
	board.print_analysis(BoardAnalysisType::AttackOpportunities, white);

	board.change("Qxc6", white);
	white = !white;
	board.print_analysis(BoardAnalysisType::PieceTypes, white);
	board.print_analysis(BoardAnalysisType::AttackOpportunities, white);

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}