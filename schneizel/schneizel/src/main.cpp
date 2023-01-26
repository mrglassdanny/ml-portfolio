#include "bitboard.h"
#include "position.h"
#include "util.h"

using namespace schneizel;

int main(int argc, char **argv)
{
	// srand(NULL);

	bitboards::init();

	auto sw = new StopWatch();

	int epochs = 50000;
	int moves = 70;

	sw->start();
	for (int j = 0; j < epochs; j++)
	{
		Position pos;
		pos.init();

		Move move;

		for (int i = 0; i < moves; i++)
		{
			auto move_list = pos.get_move_list();

			bool move_set = false;
			for (int m = 0; m < move_list.move_cnt; m++)
			{
				move = move_list.moves[m];
				if (move.gives_check)
				{
					move_set = true;
					break;
				}
			}

			if (!move_set)
				move = move_list.moves[rand() % move_list.move_cnt];

			if (pos.is_in_check(pos.white_turn))
			{
				printf("CHECK\n");
				pos.pretty_print(nullptr);
			}

			pos.make_move(move);
			pos.pretty_print(&move);
		}
	}
	sw->stop();
	sw->print_elapsed_seconds();
	printf("Moves: %d\n", epochs * moves);

	delete sw;

	return 0;
}