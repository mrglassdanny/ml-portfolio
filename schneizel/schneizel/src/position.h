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
        bitboard_t piece_bbs[(PieceTypeCnt * 2)];
        bitboard_t white_bb;
        bitboard_t black_bb;

        void init();

        bitboard_t get_all_bb();

        void get_moves();
        void make_move(Move move);
    };
}