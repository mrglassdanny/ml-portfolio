#pragma once

#include "bitboard.h"
#include "piece.h"

namespace schneizel
{
    using namespace bitboards;

    struct Move
    {
        PieceType piecetyp;
        byte_t src_sqnum;
        byte_t dst_sqnum;
        PieceType promo_piecetyp;

        Move();
        Move(PieceType piecetyp, byte_t src_sqnum, byte_t dst_sqnum);
        Move(PieceType piecetyp, byte_t src_sqnum, byte_t dst_sqnum, PieceType promo_piecetyp);
    };

    struct MoveList
    {
        Move moves[MoveMaxCnt];
        int move_cnt;
    };
}