#pragma once

#include <vector>

#include "bitboard.h"
#include "piece.h"

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
    };
}