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
        bool gives_check;

        Move();
        Move(PieceType piecetyp, square_t src_sq, square_t dst_sq);
        Move(PieceType piecetyp, square_t src_sq, square_t dst_sq, PieceType promo_piecetyp);
        Move(PieceType piecetyp, square_t src_sq, square_t dst_sq, bool gives_check);
        Move(PieceType piecetyp, square_t src_sq, square_t dst_sq, PieceType promo_piecetyp, bool gives_check);
    };

    struct PieceMoveList
    {
        bitboard_t movebb = bitboards::EmptyBB;
        bitboard_t attackbb = bitboards::EmptyBB;
        bitboard_t king_attackbb = bitboards::EmptyBB;
        bitboard_t gives_checkbb = bitboards::EmptyBB; // Move results in discovered check
    };

    struct MoveList
    {
        Move moves[MoveMaxCnt];
        int move_cnt;
    };
}