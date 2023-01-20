#include "bitboard.h"
#include "position.h"

using namespace schneizel;

int main(int argc, char **argv)
{
	bitboards::init();

	Position pos;
	pos.init();

	bitboard_t all = pos.get_all_pieces();
	bitboards::print(&all);

	return 0;
}