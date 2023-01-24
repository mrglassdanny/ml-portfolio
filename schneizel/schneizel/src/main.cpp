#include "bitboard.h"
#include "position.h"
#include "util.h"

using namespace schneizel;

int main(int argc, char **argv)
{
	srand(NULL);

	bitboards::init();

	auto sw = new StopWatch();

	int epochs = 50000;
	int moves = 50;

	sw->start();
	for (int j = 0; j < epochs; j++)
	{
		Position pos;
		pos.init();

		Move move;

		for (int i = 0; i < moves; i++)
		{
			auto move_list = pos.get_move_list();
			move = move_list.moves[rand() % move_list.move_cnt];
			pos.make_move(move);
			if (pos.is_in_check(pos.white_turn))
			{
				pos.pretty_print(&move);
				int asoidfj = 0;
			}
		}
	}
	sw->stop();
	sw->print_elapsed_seconds();
	printf("Moves: %d\n", epochs * moves);

	delete sw;

	return 0;
}