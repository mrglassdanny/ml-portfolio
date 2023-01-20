#include "bitboard.h"
#include "position.h"

using namespace schneizel;

int main(int argc, char **argv)
{
	bitboards::init();

	Position pos;
	pos.init();

	// pos.make_move(Move{9, 25});

	pos.get_moves();

	return 0;
}