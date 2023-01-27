#pragma once

#include <cctype>
#include <cstdint>

namespace schneizel
{
    typedef unsigned char byte_t;
    typedef uint64_t bitboard_t;
    typedef byte_t square_t;
    typedef byte_t row_t;
    typedef byte_t col_t;

    enum CardinalDirection
    {
        North = 0,
        East,
        South,
        West

    };

    enum DiagonalDirection
    {
        Northeast = 0,
        Southeast,
        Southwest,
        Northwest
    };
}