#pragma once

#include "bitboard.h"

namespace schneizel
{
    using namespace bitboards;

    struct Position
    {
        bool white_turn;

        bitboard_t white_pawns;
        bitboard_t white_knights;
        bitboard_t white_bishops;
        bitboard_t white_rooks;
        bitboard_t white_queens;
        bitboard_t white_king;

        bitboard_t black_pawns;
        bitboard_t black_knights;
        bitboard_t black_bishops;
        bitboard_t black_rooks;
        bitboard_t black_queens;
        bitboard_t black_king;

        void init();

        bitboard_t get_white_pieces();
        bitboard_t get_black_pieces();
        bitboard_t get_all_pieces();

        bitboard_t get_friendly_pieces(bool white);
    };
}