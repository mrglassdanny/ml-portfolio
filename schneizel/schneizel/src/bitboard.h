#pragma once

#include <iostream>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <string>

#include "constants.h"

namespace schneizel
{
    namespace bitboards
    {
        typedef unsigned char byte_t;
        typedef uint64_t bitboard_t;

        constexpr bitboard_t EmptyBB = 0ULL;
        constexpr bitboard_t FullBB = ~(EmptyBB);
        constexpr bitboard_t DarkBB = 0xAA55AA55AA55AA55ULL;
        constexpr bitboard_t LightBB = ~(DarkBB);

        constexpr bitboard_t Row1BB = 0xFF;
        constexpr bitboard_t Row2BB = Row1BB << (8 * 1);
        constexpr bitboard_t Row3BB = Row1BB << (8 * 2);
        constexpr bitboard_t Row4BB = Row1BB << (8 * 3);
        constexpr bitboard_t Row5BB = Row1BB << (8 * 4);
        constexpr bitboard_t Row6BB = Row1BB << (8 * 5);
        constexpr bitboard_t Row7BB = Row1BB << (8 * 6);
        constexpr bitboard_t Row8BB = Row1BB << (8 * 7);

        constexpr bitboard_t ColABB = 0x0101010101010101ULL;
        constexpr bitboard_t ColBBB = ColABB << 1;
        constexpr bitboard_t ColCBB = ColABB << 2;
        constexpr bitboard_t ColDBB = ColABB << 3;
        constexpr bitboard_t ColEBB = ColABB << 4;
        constexpr bitboard_t ColFBB = ColABB << 5;
        constexpr bitboard_t ColGBB = ColABB << 6;
        constexpr bitboard_t ColHBB = ColABB << 7;

        constexpr byte_t get_rownum_fr_sqnum(byte_t sqnum)
        {
            return sqnum >> 3;
        }

        constexpr bool is_rownum_valid(byte_t rownum)
        {
            return rownum >= 0 && rownum <= 7;
        }

        constexpr bitboard_t get_rowbb_fr_rownum(byte_t rownum)
        {
            return Row1BB << (8 * rownum);
        }

        constexpr bitboard_t get_rowbb_fr_sqnum(byte_t sqnum)
        {
            return get_rowbb_fr_rownum(get_rownum_fr_sqnum(sqnum));
        }

        constexpr byte_t get_colnum_fr_sqnum(byte_t sqnum)
        {
            return sqnum & 7;
        }

        constexpr bool is_colnum_valid(byte_t colnum)
        {
            return colnum >= 0 && colnum <= 7;
        }

        constexpr bitboard_t get_colbb_fr_colnum(byte_t colnum)
        {
            return ColABB << colnum;
        }

        constexpr bitboard_t get_colbb_fr_sqnum(byte_t sqnum)
        {
            return get_colbb_fr_colnum(get_colnum_fr_sqnum(sqnum));
        }

        constexpr byte_t get_sqnum(byte_t rownum, byte_t colnum)
        {
            return rownum * 8 + colnum;
        }

        constexpr byte_t get_sqval(bitboard_t bb, byte_t sqnum)
        {
            return ((bb & (1ULL << sqnum)) >> sqnum);
        }

        constexpr bitboard_t set_sqval(bitboard_t bb, byte_t sqnum)
        {
            return bb | (1ULL << sqnum);
        }

        constexpr bitboard_t clear_sqval(bitboard_t bb, byte_t sqnum)
        {
            return bb & ~(1ULL << sqnum);
        }

        constexpr bitboard_t get_sqbb(byte_t sqnum)
        {
            return 1ULL << sqnum;
        }

        struct Magic
        {
            bitboard_t maskbb;
            bitboard_t keybb;
            bitboard_t *movebbs;
            int shift;

            ~Magic();

            unsigned get_movebb_index(bitboard_t blockerbb);
        };

        void init();
        void print(bitboard_t bb);

        bitboard_t get_knight_movebb(byte_t sqnum);
        bitboard_t get_bishop_movebb(byte_t sqnum, bitboard_t bodiesbb);
        bitboard_t get_rook_movebb(byte_t sqnum, bitboard_t bodiesbb);
        bitboard_t get_queen_movebb(byte_t sqnum, bitboard_t bodiesbb);
    }
}
