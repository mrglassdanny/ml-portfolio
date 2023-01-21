#include "bitboard.h"
#include "position.h"

using namespace schneizel;

int main(int argc, char **argv)
{
	bitboards::init();

	Position pos;
	pos.init();

	Move move;

	for (int i = 0; i < 20; i++)
	{
		auto move_list = pos.get_move_list();
		move = move_list.moves[rand() % move_list.move_cnt];
		pos.make_move(move);
		pos.print(&move);
	}

	return 0;
}