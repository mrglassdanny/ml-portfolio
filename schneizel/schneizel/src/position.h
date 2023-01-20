#pragma once

#include <vector>

#include "bitboard.h"
#include "piece.h"
#include "move.h"

namespace schneizel
{
    using namespace bitboards;

    struct Position
    {
        bool white_turn;
        PieceType pieces[SquareCnt];
        bitboard_t piecebbs[(PieceTypeCnt * 2)];
        bitboard_t whitebb;
        bitboard_t blackbb;

        void init();

        bitboard_t get_allbb();

        MoveList get_move_list();
        void make_move(Move move);
    };
}