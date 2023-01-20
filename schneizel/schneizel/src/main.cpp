#include "bitboard.h"
#include "position.h"

using namespace schneizel;

int main(int argc, char **argv)
{
	bitboards::init();

	Position pos;
	pos.init();

	bitboards::print(&pos.white_bbs[PieceType::Queen]);

	return 0;
}