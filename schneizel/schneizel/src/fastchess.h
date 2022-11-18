#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

#include <windows.h>

#define ROW_CNT 8
#define COL_CNT 8
#define BOARD_LEN (ROW_CNT * COL_CNT)

#define MT ' '
#define WP 'P'
#define WN 'N'
#define WB 'B'
#define WR 'R'
#define WQ 'Q'
#define WK 'K'
#define BP 'p'
#define BN 'n'
#define BB 'b'
#define BR 'r'
#define BQ 'q'
#define BK 'k'

namespace fastchess
{
    struct Move
    {
        int src_square;
        int dst_square;
    };

    class Piece
    {
    public:
        static bool is_white(char piece);
        static bool is_black(char piece);
        static bool is_same_color(char piece_a, char piece_b);
        static const char *to_str(char piece);
    };

    struct CastleState
    {
        bool white_king_moved = false;
        bool black_king_moved = false;
        bool white_right_rook_moved = false;
        bool white_left_rook_moved = false;
        bool black_right_rook_moved = false;
        bool black_left_rook_moved = false;
    };

    class Board
    {
    private:
        char data_[BOARD_LEN];
        CastleState castle_state;
        std::vector<Move> all_moves;
        std::mutex mutx;

    public:
        static int get_row(int square);
        static int get_col(int square);
        static int get_square(int row, int col);

        static char get_alpha_col(int col);

        static bool is_row_valid(int row);
        static bool is_col_valid(int col);

        bool is_square_influenced(int square, bool by_white);

        std::vector<Move> get_diagonal_moves(int square, char piece, int row, int col);
        std::vector<Move> get_straight_moves(int square, char piece, int row, int col);

        Board();
        ~Board();

        void reset();
        void copy(Board *src);

        void print();

        char get_piece(int square);

        std::vector<Move> get_moves(int square);
        void get_moves_parallel(int square);
        std::vector<Move> get_all_moves(bool white);

        void change(Move move);
    };
}