#include "bitboard.h"
#include "position.h"
#include "util.h"

using namespace schneizel;

int main(int argc, char **argv)
{
	// srand(NULL);

	bitboards::init();

	auto sw = new StopWatch();

	int game_cnt = 50000;
	int move_cnt = 50;

	sw->start();
	for (int game_num = 0; game_num < game_cnt; game_num++)
	{
		Position pos;
		pos.init();

		Move move;

		for (int move_num = 0; move_num < move_cnt; move_num++)
		{

			auto move_list = pos.get_move_list();
			if (move_list.move_cnt == 0)
			{
				printf("CHECKMATE\n");
				pos.pretty_print(&move);
				printf("Game: %d\n", game_num);
				break;
			}

			if (pos.is_in_check(pos.white_turn))
			{
				printf("CHECK\n");
				pos.pretty_print(&move);
				printf("Game: %d\tLegal moves: %d\n", game_num, move_list.move_cnt);
			}

			move = move_list.moves[rand() % move_list.move_cnt];
			pos.make_move(move);
		}
	}
	sw->stop();
	sw->print_elapsed_seconds();
	printf("Moves: %d\n", game_cnt * move_cnt);

	delete sw;

	return 0;
}