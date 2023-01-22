#pragma once

#include <vector>
#include <windows.h>

#include "bitboard.h"
#include "piece.h"
#include "move.h"

namespace schneizel
{
    using namespace bitboards;

    struct CastleRights
    {
        bool white_left = true;
        bool white_right = true;
        bool black_left = true;
        bool black_right = true;
    };

    struct Position
    {
        bool white_turn;
        PieceType pieces[SquareCnt];
        bitboard_t piecebbs[(PieceTypeCnt * 2)];
        bitboard_t whitebb;
        bitboard_t blackbb;
        CastleRights castle_rights;
        byte_t au_passant_sqnum = 0;

        void init();
        void pretty_print(Move *prev_move);

        bitboard_t get_allbb();

        MoveList get_move_list();
        void make_move(Move move);
    };
}