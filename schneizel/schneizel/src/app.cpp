#include <stdio.h>

#include <ATen/ATen.h>

#include "chess.h"

using namespace chess;

int main()
{
	srand(time(NULL));

	MoveOpportunities opps;

	auto board = Openings::create(OpeningType::GrunfeldDefense);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);
	board.print(BoardAnalysisType::AttackOpportunities);

	board.change("Qa4", true);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);
	board.print(BoardAnalysisType::AttackOpportunities);

	board.change("b5", false);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);
	board.print(BoardAnalysisType::AttackOpportunities);

	board.change("Qxb5", true);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);
	board.print(BoardAnalysisType::AttackOpportunities);

	board.change("c6", false);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);
	board.print(BoardAnalysisType::AttackOpportunities);

	board.change("Qb7", true);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);
	board.print(BoardAnalysisType::AttackOpportunities);

	board.change("Na6", false);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);
	board.print(BoardAnalysisType::AttackOpportunities);

	board.change("Qxc6", true);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);
	board.print(BoardAnalysisType::AttackOpportunities);
	board.print_move_opportunities(&opps);

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}