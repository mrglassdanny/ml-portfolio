#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

#include <windows.h>

#define CHESS_THROW_ERROR(chess_err_msg) \
    printf("%s", chess_err_msg);         \
    exit(1)

#define CHESS_ROW_CNT 8
#define CHESS_COL_CNT 8
#define CHESS_BOARD_LEN (CHESS_ROW_CNT * CHESS_COL_CNT)

#define CHESS_MT ' '
#define CHESS_WP 'P'
#define CHESS_WN 'N'
#define CHESS_WB 'B'
#define CHESS_WR 'R'
#define CHESS_WQ 'Q'
#define CHESS_WK 'K'
#define CHESS_BP 'p'
#define CHESS_BN 'n'
#define CHESS_BB 'b'
#define CHESS_BR 'r'
#define CHESS_BQ 'q'
#define CHESS_BK 'k'

#define CHESS_INVALID_SQUARE -1

#define CHESS_WHITE_CHECKMATED_VAL -1000
#define CHESS_BLACK_CHECKMATED_VAL 1000

#define CHESS_OPENING_MOVE_CNT 8

namespace chess
{
    struct Move
    {
        int src_square;
        int dst_square;
        char promo_piece = CHESS_MT;

        static bool is_valid(Move *move);
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

    struct Simulation;
    struct Evaluation;
    struct EvaluationData;

    class Board
    {
    private:
        char data_[CHESS_BOARD_LEN];

        struct CastleState
        {
            bool white_king_moved = false;
            bool black_king_moved = false;
            bool white_right_rook_moved = false;
            bool white_left_rook_moved = false;
            bool black_right_rook_moved = false;
            bool black_left_rook_moved = false;
        } castle_state_;

        struct CheckState
        {
            bool white_checked;
            bool black_checked;
            bool white_king_pins[CHESS_BOARD_LEN];
            bool black_king_pins[CHESS_BOARD_LEN];
        } check_state_;

        struct AuPassantState
        {
            int dst_col = CHESS_INVALID_SQUARE;
        } au_passant_state_;

        static bool is_row_valid(int row);
        static bool is_col_valid(int col);

        void update_diagonal_pins(int square);
        void update_straight_pins(int square);
        void update_pins();

        std::vector<Move> get_diagonal_moves(int square, char piece, int row, int col);
        std::vector<Move> get_straight_moves(int square, char piece, int row, int col);

        bool is_square_under_attack(int square, bool by_white);

        bool is_piece_in_king_pin(int square, bool white_king_pin);

        static Evaluation sim_minimax_alphabeta_sync(Simulation sim, bool white, int depth, int max_depth, int depth_inc, int max_depth_inc, int depth_inc_max_move_cnt, int alpha, int beta);
        static void sim_minimax_alphabeta_async(Simulation sim, bool white, int depth, int depth_inc, int depth_inc_max_move_cnt, int alpha, int beta, EvaluationData *evals);

    public:
        Board();
        ~Board();

        static int get_row(int square);
        static int get_row(char alpha_row);
        static int get_col(int square);
        static int get_col(char alpha_col);
        static char get_alpha_col(int col);
        static int get_square(int row, int col);
        static int get_square(int row, char alpha_col);
        static int get_square(char alpha_row, char alpha_col);

        void reset();

        void copy(Board *src);

        char *get_data();

        int compare_data(Board *other);
        int compare_data(const char *other_data);

        void print();
        void print(Move move);

        char get_piece(int square);

        std::vector<Move> get_moves(int square, bool test_check);
        std::vector<Move> get_all_moves(bool white);
        std::vector<Move> get_all_moves(bool white, std::vector<int> src_squares);
        bool has_moves(bool white);

        int get_king_square(bool white);

        bool is_check(bool by_white);
        bool is_check(bool by_white, bool hard_way);
        bool is_checkmate(bool by_white);
        bool is_checkmate(bool by_white, bool hard_way);

        void change(Move move);
        Move change(std::string move_str, bool white);

        Simulation simulate(Move move);
        Simulation simulate(std::string move_str, bool white);
        std::vector<Simulation> simulate_all(bool white);
        std::vector<Simulation> simulate_all(bool white, std::vector<int> src_squares);

        int evaluate_material();

        std::vector<EvaluationData> minimax_alphabeta(bool white, int depth, int depth_inc, int depth_inc_max_move_cnt);
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
        int depth;
    };

    struct EvaluationData
    {
        Evaluation eval;
        Move move;
        Board board;
    };

    struct Opening
    {
        char boards[CHESS_BOARD_LEN * CHESS_OPENING_MOVE_CNT];
        char move_strs[64];
        int game_cnt = 0;
    };

    class OpeningEngine
    {
    private:
        std::vector<Opening> openings_;

        static bool sort_fn(Opening const &a, Opening const &b);

    public:
        OpeningEngine(const char *opening_path);
        ~OpeningEngine();

        std::string next_move(Board *board, int move_cnt);
    };

    struct PGNGame
    {
        int lbl;
        std::vector<std::string> move_strs;
    };

    class PGN
    {
    public:
        static std::vector<PGNGame *> import(const char *path, long long file_size);
        static void export_openings(const char *pgn_path, long long pgn_file_size, const char *export_path);
    };
}