#pragma once

#include <iostream>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <string>

#include "constants.h"

typedef unsigned char byte_t;
typedef uint64_t bitboard_t;

constexpr bitboard_t Empty = 0ULL;
constexpr bitboard_t Full = ~(Empty);
constexpr bitboard_t Dark = 0xAA55AA55AA55AA55ULL;
constexpr bitboard_t Light = ~(Dark);

constexpr bitboard_t Row1 = 0xFF;
constexpr bitboard_t Row2 = Row1 << (8 * 1);
constexpr bitboard_t Row3 = Row1 << (8 * 2);
constexpr bitboard_t Row4 = Row1 << (8 * 3);
constexpr bitboard_t Row5 = Row1 << (8 * 4);
constexpr bitboard_t Row6 = Row1 << (8 * 5);
constexpr bitboard_t Row7 = Row1 << (8 * 6);
constexpr bitboard_t Row8 = Row1 << (8 * 7);

constexpr bitboard_t ColA = 0x0101010101010101ULL;
constexpr bitboard_t ColB = ColA << 1;
constexpr bitboard_t ColC = ColA << 2;
constexpr bitboard_t ColD = ColA << 3;
constexpr bitboard_t ColE = ColA << 4;
constexpr bitboard_t ColF = ColA << 5;
constexpr bitboard_t ColG = ColA << 6;
constexpr bitboard_t ColH = ColA << 7;

constexpr int get_rownum_fr_sqnum(int sqnum)
{
    return sqnum >> 3;
}

constexpr bool is_rownum_valid(int rownum)
{
    return rownum >= 0 && rownum <= 7;
}

constexpr bitboard_t get_row_fr_rownum(int rownum)
{
    return Row1 << (8 * rownum);
}

constexpr bitboard_t get_row_fr_sqnum(int sqnum)
{
    return get_row_fr_rownum(get_rownum_fr_sqnum(sqnum));
}

constexpr int get_colnum_fr_sqnum(int sqnum)
{
    return sqnum & 7;
}

constexpr bool is_colnum_valid(int colnum)
{
    return colnum >= 0 && colnum <= 7;
}

constexpr bitboard_t get_col_fr_colnum(int colnum)
{
    return ColA << colnum;
}

constexpr bitboard_t get_col_fr_sqnum(int sqnum)
{
    return get_col_fr_colnum(get_colnum_fr_sqnum(sqnum));
}

constexpr int get_sqnum(int rownum, int colnum)
{
    return rownum * 8 + colnum;
}

constexpr int get_sqval(bitboard_t bb, int sqnum)
{
    return ((bb & (1ULL << sqnum)) >> sqnum);
}

constexpr bitboard_t get_sq(int sqnum)
{
    return 1ULL << sqnum;
}

constexpr bitboard_t set_sq(bitboard_t bb, int sqnum)
{
    return bb | (1ULL << sqnum);
}

struct Magic
{
    bitboard_t mask;
    bitboard_t key;
    bitboard_t *attacks;
    int shift;

    ~Magic();

    unsigned get_attack_index(bitboard_t occupied);
};

class MagicPRNG
{
    uint64_t s;
    uint64_t rand64()
    {
        s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
        return s * 2685821657736338717LL;
    }

public:
    MagicPRNG(uint64_t seed) : s(seed) { assert(seed); }

    bitboard_t rand() { return (bitboard_t)rand64(); }

    bitboard_t sparse_rand()
    {
        return (bitboard_t)(rand64() & rand64() & rand64());
    }
};

void init();
void print(bitboard_t *bb);
void pretty_print(bitboard_t *bb);