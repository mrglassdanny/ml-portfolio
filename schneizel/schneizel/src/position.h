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
        Piece white_pieces[PieceMaxCnt];
        Piece black_pieces[PieceMaxCnt];
        bitboard_t white_bbs[PieceTypeCnt];
        bitboard_t black_bbs[PieceTypeCnt];

        void init();

        bitboard_t get_whitebb();
        bitboard_t get_blackbb();
        bitboard_t get_allbb();

        Move get_moves();
        void make_move(Move move);
    };
}