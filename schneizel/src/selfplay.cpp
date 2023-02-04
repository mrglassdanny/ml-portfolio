#include "selfplay.h"

using namespace Stockfish;

namespace Selfplay
{
    struct PostGamePosition
    {
        std::string fen;
        int eval = 0;
        int delta = 0;
    };

    void analyze_game(std::vector<PostGamePosition> postgame_positions)
    {
        Position pos;
        StateListPtr states(new std::deque<StateInfo>(1));

        for (auto p : postgame_positions)
        {
            states = StateListPtr(new std::deque<StateInfo>(1));
            pos.set(p.fen, false, &states->back(), Threads.main());
            printf("\n\nEval: %d\tDelta: %d\n\n", p.eval, p.delta);
            std::cout << pos;
        }
    }

    void play_game(int depth)
    {
        Position pos;
        StateListPtr states(new std::deque<StateInfo>(1));
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                false, &states->back(), Threads.main());

        std::vector<PostGamePosition> postgame_positions;
        int delta = 0;
        int prev_eval = 0;

        int move_cnt = 0;
        while (true)
        {
            Search::LimitsType limits;
            {
                limits.startTime = now();

                if (move_cnt % 2 == 0)
                    limits.depth = depth;
                else
                    limits.depth = 1;
            }

            MoveList<LEGAL> move_list(pos);
            if (move_list.size() == 0)
            {
                if (pos.checkers() != 0ULL)
                {
                    printf("\n ====================== CHECKMATE ======================\n");
                }
                else
                {
                    printf("\n ====================== STALEMATE ======================\n");
                }

                break;
            }

            for (auto move : move_list)
            {
                limits.searchmoves.push_back(move);
            }

            Threads.start_thinking(pos, states, limits, false);
            Threads.main()->wait_for_search_finished();
            auto best_thread = Threads.get_best_thread();
            auto best_move = best_thread->rootMoves[0].pv[0];
            auto best_move_str = UCI::move(best_move, false);

            states = StateListPtr(new std::deque<StateInfo>(1));
            pos.set(pos.fen(), false, &states->back(), Threads.main());
            states->emplace_back();
            pos.do_move(best_move, states->back());
            int eval = pos.material_eval();
            delta = eval - prev_eval;
            prev_eval = eval;
            postgame_positions.push_back(PostGamePosition{pos.fen(), eval, delta});

            printf("MATERIAL: %d\n", eval);

            if (pos.checkers())
            {
                printf("\n ====================== CHECK ======================\n");
            }
            else
            {
                printf("STOCKFISH: %d\n", Eval::evaluate(pos));
            }

            std::cout << pos;

            move_cnt++;
        }

        analyze_game(postgame_positions);
    }

    void loop()
    {
        play_game(10);
    }

}