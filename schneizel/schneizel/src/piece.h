#pragma once

#include "constants.h"
#include "bitboard.h"

namespace schneizel
{
    using namespace bitboards;

    enum PieceType : byte_t
    {
        WhitePawn,
        WhiteKnight,
        WhiteBishop,
        WhiteRook,
        WhiteQueen,
        WhiteKing,
        BlackPawn,
        BlackKnight,
        BlackBishop,
        BlackRook,
        BlackQueen,
        BlackKing,
        None
    };

    char get_piecetyp_char(PieceType piecetyp);
    const char *get_piecetyp_str(PieceType piecetyp);
}