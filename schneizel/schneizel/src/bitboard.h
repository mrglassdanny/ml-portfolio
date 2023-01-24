#pragma once

#include <iostream>
#include <cstdlib>
#include <string>

#include "constant.h"
#include "type.h"

namespace schneizel
{
    namespace bitboards
    {
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

        constexpr row_t get_row_fr_sq(square_t sq)
        {
            return sq >> 3;
        }

        constexpr bool is_row_valid(row_t row)
        {
            return row >= 0 && row <= 7;
        }

        constexpr bitboard_t get_rowbb_fr_row(row_t row)
        {
            return Row1BB << (8 * row);
        }

        constexpr bitboard_t get_rowbb_fr_sq(square_t sq)
        {
            return get_rowbb_fr_row(get_row_fr_sq(sq));
        }

        constexpr col_t get_col_fr_sq(square_t sq)
        {
            return sq & 7;
        }

        constexpr bool is_col_valid(col_t col)
        {
            return col >= 0 && col <= 7;
        }

        constexpr bitboard_t get_colbb_fr_col(col_t col)
        {
            return ColABB << col;
        }

        constexpr bitboard_t get_colbb_fr_sq(square_t sq)
        {
            return get_colbb_fr_col(get_col_fr_sq(sq));
        }

        constexpr square_t get_sq(row_t row, col_t col)
        {
            return row * 8 + col;
        }

        constexpr byte_t get_sqval(bitboard_t bb, square_t sq)
        {
            return ((bb & (1ULL << sq)) >> sq);
        }

        constexpr bitboard_t set_sqval(bitboard_t bb, square_t sq)
        {
            return bb | (1ULL << sq);
        }

        constexpr bitboard_t clear_sqval(bitboard_t bb, square_t sq)
        {
            return bb & ~(1ULL << sq);
        }

        constexpr bitboard_t get_sqbb(square_t sq)
        {
            return 1ULL << sq;
        }

        inline square_t lsb(bitboard_t bb)
        {
            unsigned long idx;
            _BitScanForward64(&idx, bb);
            return (square_t)idx;
        }

        inline square_t msb(bitboard_t bb)
        {
            unsigned long idx;
            _BitScanReverse64(&idx, bb);
            return (square_t)idx;
        }

        inline square_t pop_lsb(bitboard_t &bb)
        {
            const square_t sq = lsb(bb);
            bb &= bb - 1;
            return sq;
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
        void print(bitboard_t bb, square_t sq);

        bitboard_t get_knight_movebb(square_t sq);
        bitboard_t get_bishop_movebb(square_t sq, bitboard_t bodiesbb);
        bitboard_t get_rook_movebb(square_t sq, bitboard_t bodiesbb);
        bitboard_t get_queen_movebb(square_t sq, bitboard_t bodiesbb);
        bitboard_t get_king_movebb(square_t sq);

        Magic *get_bishop_magic(square_t sq);
        Magic *get_rook_magic(square_t sq);
    }
}
