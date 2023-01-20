#pragma once

#include "bitboard.h"
#include "piece.h"

namespace schneizel
{
    using namespace bitboards;

    struct Move
    {
        byte_t src_sqnum;
        byte_t dst_sqnum;
    };

    struct MoveList
    {
        Move moves[MoveMaxCnt];
    };
}