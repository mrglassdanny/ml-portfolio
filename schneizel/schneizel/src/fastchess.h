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

    struct Evaluation
    {
        float value;
        Move move;
    };

    class Piece
    {
    public:
        static bool is_white(char piece);
        static bool is_black(char piece);
        static bool is_same_color(char piece_a, char piece_b);
        static const char *to_str(char piece);
        static int get_value(char piece);
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

    struct Simulation;

    class Board
    {
    private:
        char data_[BOARD_LEN];
        CastleState castle_state_;

        static int get_row(int square);
        static int get_col(int square);
        static char get_alpha_col(int col);
        static int get_square(int row, int col);
        static bool is_row_valid(int row);
        static bool is_col_valid(int col);

        bool is_square_under_attack(int square, bool by_white);
        bool is_check(bool by_white);

        std::vector<Move> get_diagonal_moves(int square, char piece, int row, int col);
        std::vector<Move> get_straight_moves(int square, char piece, int row, int col);

        static float sim_minimax_sync(Simulation sim, bool white, int depth, float alpha, float beta);
        static void sim_minimax_async(Simulation sim, bool white, int depth, float alpha, float beta, Evaluation *evals);

    public:
        Board();
        ~Board();

        void reset();
        void copy(Board *src);

        void print();

        char get_piece(int square);

        std::vector<Move> get_moves(int square, bool test_check);
        std::vector<Move> get_all_moves(bool white);

        void change(Move move);
        void change_random(bool white);

        Simulation simulate(Move move);
        std::vector<Simulation> simulate_all(bool white);

        int evaluate_material();

        void change_minimax_sync(bool white, int depth);
        void change_minimax_async(bool white, int depth);
    };

    struct Simulation
    {
        int idx = -1;
        Move move;
        Board board;
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