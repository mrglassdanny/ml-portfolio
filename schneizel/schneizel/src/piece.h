#pragma once

#include "constants.h"
#include "bitboard.h"

namespace schneizel
{
    using namespace bitboards;

    enum PieceType : byte_t
    {
        Pawn,
        Knight,
        Bishop,
        Rook,
        Queen,
        King,
        None
    };

    struct Piece
    {
        PieceType typ;
        byte_t sqnum;
    };
}