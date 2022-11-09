#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <iostream>
#include <vector>

#define CHESS_BOARD_ROW_CNT 8
#define CHESS_BOARD_COL_CNT 8
#define CHESS_BOARD_LEN (CHESS_BOARD_COL_CNT * CHESS_BOARD_ROW_CNT)

#define CHESS_MOVE_LIMIT 500

#define CHESS_INVALID_VALUE -1

#define CHESS_ONE_HOT_ENCODE_COMBINATION_CNT 6
#define CHESS_ONE_HOT_ENCODED_BOARD_LEN (CHESS_BOARD_LEN * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT)

namespace chess
{
    struct Move
    {
        int src_idx;
        int dst_idx;
    };

    struct MinimaxResult
    {
        float eval;
        bool prune_flg;
    };

    enum PieceType
    {
        Empty = 0,
        WhitePawn = 1,
        WhiteKnight = 3,
        WhiteBishop = 4,
        WhiteRook = 6,
        WhiteQueen = 9,
        WhiteKing = 10,
        BlackPawn = -1,
        BlackKnight = -3,
        BlackBishop = -4,
        BlackRook = -6,
        BlackQueen = -9,
        BlackKing = -10
    };

    class Piece
    {
    public:
        static PieceType get_piece_fr_char(char piece_id, bool white);
        static char get_char_fr_piece(PieceType piece);
        static bool is_piece_white(PieceType piece);
        static bool is_piece_black(PieceType piece);
        static bool is_piece_same_color(PieceType a, PieceType b);
        static float piece_to_float(PieceType piece);
    };

    enum BoardStatus
    {
        Normal,
        WhiteInCheck,
        BlackInCheck,
        WhiteInCheckmate,
        BlackInCheckmate,
        WhiteInStalemate,
        BlackInStalemate,
        WhiteInsufficientMaterial,
        BlackInsufficientMaterial,
        MoveLimitExceeded
    };

    class Board
    {
    private:
        bool white_;
        int move_cnt_;
        int data_[CHESS_BOARD_LEN];

        static int get_col_fr_adj_col(int adj_col);
        static int get_adj_col_fr_col(char col);
        static int get_row_fr_char(char row);
        static int get_adj_row_fr_row(int row);
        static int get_adj_col_fr_idx(int idx);
        static int get_adj_row_fr_idx(int idx);
        static char get_col_fr_idx(int idx);
        static int get_row_fr_idx(int idx);
        static int get_idx_fr_colrow(char col, int row);
        static int get_idx_fr_adj_colrow(int adj_col, int adj_row);
        static bool is_row_valid(int row);
        static bool is_adj_colrow_valid(int adj_col, int adj_row);

        bool is_square_under_attack(int idx);

        void get_piece_straight_moves(PieceType piece, int adj_col, int adj_row, int cnt, std::vector<int> *out);
        void get_piece_straight_influence(PieceType piece, int adj_col, int adj_row, int cnt, std::vector<int> *out);
        void get_piece_diagonal_moves(PieceType piece, int adj_col, int adj_row, int cnt, std::vector<int> *out);
        void get_piece_diagonal_influence(PieceType piece, int adj_col, int adj_row, int cnt, std::vector<int> *out);

    public:
        Board();
        ~Board();

        bool operator==(const Board &);
        bool operator!=(const Board &);

        void reset();
        void copy(Board *src);

        void print();
        void print_flipped();

        bool is_white_turn();

        std::vector<int> get_piece_moves(int piece_idx, bool test_in_check_flg);
        std::vector<int> get_piece_influence(int piece_idx);

        Move get_random_move();
        std::string convert_move_to_an_move(Move move);

        Move change(std::string an_move);
        Move change(Move move);
        Move change();
        Board simulate(Move move);
        std::vector<Board> simulate_all_moves();

        bool check();
        bool check(bool white);
        bool checkmate();
        bool checkmate(bool white);
        bool stalemate();
        bool stalemate(bool white);
        bool insufficient_material();
        bool insufficient_material(bool white);
        bool game_over();

        BoardStatus get_status();
        void print_status();

        void one_hot_encode(int *out);
        void one_hot_encode(float *out);
    };
}
