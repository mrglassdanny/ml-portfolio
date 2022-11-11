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

	auto board = Openings::create(OpeningType::SlavDefense);
	board.print(BoardAnalysisType::PieceTypes);
	board.print(BoardAnalysisType::Material);
	board.print(BoardAnalysisType::Influence);

	for (int i = 0; i < 3; i++)
	{
		auto move = board.convert_move_to_an_move(board.get_random_move());
		printf("%s\n", move.c_str());
		board.change(move);
		board.print(BoardAnalysisType::PieceTypes);
		board.print(BoardAnalysisType::Material);
		board.print(BoardAnalysisType::Influence);
	}

	// at::Tensor a = at::ones({2, 2}, at::kInt);
	// at::Tensor b = at::randn({2, 2});
	// auto c = a + b.to(at::kInt);

	return 0;
}