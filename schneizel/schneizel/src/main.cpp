#include "bitboard.h"
#include "position.h"

using namespace schneizel;

int main(int argc, char **argv)
{
	bitboards::init();

	Position pos;
	pos.init();

	pos.make_move(Move{PieceType::Pawn, 8, 16});

	bitboard_t all = pos.get_allbb();

	bitboards::print(&all);

	return 0;
}