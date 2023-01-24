#pragma once

#include "bitboard.h"
#include "piece.h"

namespace schneizel
{
    using namespace bitboards;

    struct Move
    {
        PieceType piecetyp;
        square_t src_sq;
        square_t dst_sq;
        PieceType promo_piecetyp;

        Move();
        Move(PieceType piecetyp, square_t src_sq, square_t dst_sq);
        Move(PieceType piecetyp, square_t src_sq, square_t dst_sq, PieceType promo_piecetyp);
    };

    struct PieceMoveList
    {
        bitboard_t movebb;
        bitboard_t attackbb;
    };

    struct MoveList
    {
        Move moves[MoveMaxCnt];
        int move_cnt;
    };
}