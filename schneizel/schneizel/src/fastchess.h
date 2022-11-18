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

    struct SimEval
    {
        char board_data[BOARD_LEN];
        float eval;
    };

    class Piece
    {
    public:
        static bool is_white(char piece);
        static bool is_black(char piece);
        static bool is_same_color(char piece_a, char piece_b);
        static const char *to_str(char piece);
        static float get_value(char piece);
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
        CastleState castle_state_;
        std::vector<Move> all_moves_;
        std::mutex mutx_;

        static int get_row(int square);
        static int get_col(int square);
        static char get_alpha_col(int col);
        static int get_square(int row, int col);
        static bool is_row_valid(int row);
        static bool is_col_valid(int col);

        bool is_square_influenced(int square, bool by_white);

        std::vector<Move> get_diagonal_moves(int square, char piece, int row, int col);
        std::vector<Move> get_straight_moves(int square, char piece, int row, int col);

        void get_moves_parallel(int square);

    public:
        Board();
        ~Board();

        void reset();
        void copy(Board *src);

        void print();

        char get_piece(int square);

        std::vector<Move> get_moves(int square);
        std::vector<Move> get_all_moves(bool white);

        void change(Move move);
        void change_random(bool white);

        Board *simulate(Move move);
        std::vector<Board *> simulate_all(bool white);

        float evaluate_material();

        float minimax(bool white, int depth, float alpha, float beta);
        void minimax_parallel(bool white, int depth, float alpha, float beta, std::vector<SimEval> *sim_evals, std::mutex *mutx);
        void change_minimax(bool white, int depth);
    };

    class StopWatch
    {
    public:
        virtual void start() = 0;
        virtual void stop() = 0;

        virtual double get_elapsed_seconds() = 0;
        virtual void print_elapsed_seconds() = 0;
    };

    class CpuStopWatch : public StopWatch
    {
    private:
        clock_t beg_;
        clock_t end_;

    public:
        CpuStopWatch();
        ~CpuStopWatch();

        virtual void start();
        virtual void stop();

        virtual double get_elapsed_seconds();
        virtual void print_elapsed_seconds();
    };
}