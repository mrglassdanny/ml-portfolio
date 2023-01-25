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
        bitboard_t white_attackbbs[SquareCnt];
        bitboard_t white_attackbb;
        bitboard_t black_attackbbs[SquareCnt];
        bitboard_t blackbb;
        bitboard_t black_attackbb;
        bitboard_t allbb;
        CastleRights castle_rights;
        square_t au_passant_sq = 0;

        void init();
        void pretty_print(Move *prev_move);

        PieceMoveList get_white_pawn_moves(square_t src_sq);
        PieceMoveList get_white_knight_moves(square_t src_sq);
        PieceMoveList get_white_bishop_moves(square_t src_sq);
        PieceMoveList get_white_rook_moves(square_t src_sq);
        PieceMoveList get_white_queen_moves(square_t src_sq);
        PieceMoveList get_white_king_moves(square_t src_sq);
        PieceMoveList get_black_pawn_moves(square_t src_sq);
        PieceMoveList get_black_knight_moves(square_t src_sq);
        PieceMoveList get_black_bishop_moves(square_t src_sq);
        PieceMoveList get_black_rook_moves(square_t src_sq);
        PieceMoveList get_black_queen_moves(square_t src_sq);
        PieceMoveList get_black_king_moves(square_t src_sq);

        bool is_in_check(bool white);

        MoveList get_move_list();
        void make_move(Move move);
    };
}