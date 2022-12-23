#pragma once

#include <iostream>
#include <vector>
#include <thread>

#include <windows.h>

#include <zero/mod.cuh>

#define CHESS_THROW_ERROR(chess_err_msg) \
    printf("%s", chess_err_msg);         \
    exit(1)

#define CHESS_ROW_CNT 8
#define CHESS_COL_CNT 8
#define CHESS_BOARD_LEN (CHESS_ROW_CNT * CHESS_COL_CNT)

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

#define CHESS_INVALID_SQUARE -1

#define CHESS_EVAL_MIN_VAL -1000
#define CHESS_EVAL_MAX_VAL 1000

#define CHESS_BOARD_CHANNEL_CNT 6

namespace chess
{
    struct Move
    {
        int src_square;
        int dst_square;
        char promo_piece = MT;
    };

    class Piece
    {
    public:
        static bool is_white(char piece);
        static bool is_black(char piece);
        static bool is_same_color(char piece_a, char piece_b);
        static const char *to_str(char piece);
        static int get_value(char piece);
        static char get_pgn_piece(char piece);
        static char get_piece_fr_pgn_piece(char pgn_piece, bool white);
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

    struct CheckState
    {
        bool white_checked;
        bool black_checked;
        bool white_king_pins[CHESS_BOARD_LEN];
        bool black_king_pins[CHESS_BOARD_LEN];
    };

    struct AuPassantState
    {
        int dst_col = CHESS_INVALID_SQUARE;
    };

    struct Simulation;
    struct Evaluation;

    class Board
    {
    private:
        char data_[CHESS_BOARD_LEN];
        CastleState castle_state_;
        CheckState check_state_;
        AuPassantState au_passant_state_;

        static int get_row(int square);
        static int get_row(char alpha_row);
        static int get_col(int square);
        static int get_col(char alpha_col);
        static char get_alpha_col(int col);
        static int get_square(int row, int col);
        static int get_square(int row, char alpha_col);
        static int get_square(char alpha_row, char alpha_col);
        static bool is_row_valid(int row);
        static bool is_col_valid(int col);

        std::vector<Move> get_diagonal_moves(int square, char piece, int row, int col);
        std::vector<Move> get_straight_moves(int square, char piece, int row, int col);

        static int sim_minimax_alphabeta_sync(Simulation sim, bool white, int depth, int alpha, int beta);
        static void sim_minimax_alphabeta_async(Simulation sim, bool white, int depth, int alpha, int beta, Evaluation *evals);

        static int sim_dyn_minimax_alphabeta_dyn_sync(Simulation sim, bool white, int depth, int alpha, int beta, int depth_inc_cnt);
        static void sim_minimax_alphabeta_dyn_async(Simulation sim, bool white, int depth, int alpha, int beta, Evaluation *evals);

    public:
        Board();
        ~Board();

        void reset();
        void copy(Board *src);

        void print();
        void print(Move move);

        char *get_data();

        char get_piece(int square);
        int get_king_square(bool white);
        bool is_piece_in_king_pin(int square, bool white_king_pin);

        std::vector<Move> get_moves(int square, bool test_check);
        std::vector<Move> get_all_moves(bool white);
        void update_pins(bool white);

        // NOTE: this will only work if invoked BEFORE move is made to board!
        std::string convert_move_to_move_str(Move move);

        bool has_moves(bool white);
        bool is_square_under_attack(int square, bool by_white);
        bool is_check(bool by_white, bool hard_way);
        bool is_checkmate(bool by_white, bool hard_way);

        void change(Move move);
        Move change(std::string move_str, bool white);

        Simulation simulate(Move move);
        std::vector<Simulation> simulate_all(bool white);

        int evaluate_material();

        std::vector<Evaluation> minimax_alphabeta(bool white, int depth);
        std::vector<Evaluation> minimax_alphabeta_dyn(bool white, int depth);

        void material_encode(float *out);
        void one_hot_encode(float *out);
        void one_hot_encode_w_moves(float *out, bool white);
    };

    struct Simulation
    {
        int idx = -1;
        Move move;
        Board board;
    };

    struct Evaluation
    {
        int value;
        Move move;
        Board board;
    };

    struct PGNGame
    {
        int lbl;
        std::vector<std::string> move_strs;
    };

    class PGN
    {
    public:
        static std::vector<PGNGame *> import(const char *path);
    };
}