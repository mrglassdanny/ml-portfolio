#include <stdio.h>

#include <ATen/ATen.h>

#include "chess.h"

using namespace chess;

int main()
{
	srand(time(NULL));

	// Board board;

	// while (!board.game_over())
	// {
	// 	printf("\n================================= =================================\n");
	// 	board.print_status();
	// 	board.change();
	// 	board.print();
	// 	board.print_status();
	// 	printf("================================= =================================\n");
	// }

	auto board = Openings::create(OpeningType::GrunfeldDefense);
	board.print(BoardAnalysisType::PieceTypes);

	bool white = true;
	// for (int i = 0; i < 10; i++)
	// {
	// 	auto move = board.convert_move_to_an_move(board.get_random_move(white));
	// 	printf("%s\n", move.c_str());
	// 	board.change(move, white);
	// 	board.print(BoardAnalysisType::PieceTypes);

	// 	white = !white;
	// }

	board.change("Qa4", white);
	board.pretty_print();
	// board.print(BoardAnalysisType::Material);
	// board.print(BoardAnalysisType::Influence);
	// board.print(BoardAnalysisType::MaterialInfluencePieceWise);

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}