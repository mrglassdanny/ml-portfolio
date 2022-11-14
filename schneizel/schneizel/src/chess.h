#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <iostream>
#include <vector>

#include <windows.h>

#define CHESS_BOARD_ROW_CNT 8
#define CHESS_BOARD_COL_CNT 8
#define CHESS_BOARD_LEN (CHESS_BOARD_COL_CNT * CHESS_BOARD_ROW_CNT)

#define CHESS_MOVE_LIMIT 250

#define CHESS_INVALID_VALUE -1

#define CHESS_ONE_HOT_ENCODE_COMBINATION_CNT 6
#define CHESS_ONE_HOT_ENCODED_BOARD_LEN (CHESS_BOARD_LEN * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT)

#define CHESS_OPENING_E4 1000
#define CHESS_OPENING_E4_CNT 13
#define CHESS_OPENING_D4 2000
#define CHESS_OPENING_D4_CNT 13
#define CHESS_OPENING_OTHER 3000
#define CHESS_OPENING_OTHER_CNT 12

namespace chess
{
    enum PieceType
    {
        Empty = 0,
        WhitePawn = 1,
        WhiteKnight = 2,
        WhiteBishop = 3,
        WhiteRook = 4,
        WhiteQueen = 5,
        WhiteKing = 6,
        BlackPawn = -1,
        BlackKnight = -2,
        BlackBishop = -3,
        BlackRook = -4,
        BlackQueen = -5,
        BlackKing = -6
    };

    class Piece
    {
    public:
        static PieceType fr_char(char piece_id, bool white);
        static char fr_piece(PieceType piece);
        static bool is_white(PieceType piece);
        static bool is_black(PieceType piece);
        static bool is_same_color(PieceType a, PieceType b);
        static int to_int(PieceType typ);
        static const char *to_str(PieceType typ);
        static const char *to_pretty_str(PieceType typ);
    };

    struct Move
    {
        int src_idx;
        int dst_idx;
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
        BlackInsufficientMaterial
    };

    enum BoardAnalysisType
    {
        PieceTypes,
        Material,
        Influence
    };

    class Board
    {
    private:
        int data_[CHESS_BOARD_LEN];
        int material_data_[CHESS_BOARD_LEN];
        int influence_data_[CHESS_BOARD_LEN];
        int attack_opportunities_data_[CHESS_BOARD_LEN];

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

        bool is_square_under_attack(int idx, bool white);

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

        static void print(int *board);
        void print();
        void pretty_print();
        void print_analysis(BoardAnalysisType typ);

        std::vector<int> get_piece_moves(int piece_idx, bool test_check);
        std::vector<int> get_piece_influence(int piece_idx, bool test_check);

        Move get_random_move(bool white);
        std::string convert_move_to_an_move(Move move);

        Move change(std::string an_move, bool white);
        Move change(Move move, bool white);
        Move change(bool white);
        Board simulate(Move move, bool white);
        std::vector<Board> simulate_all_moves(bool white);

        bool check(bool white);
        bool checkmate(bool white);
        bool stalemate(bool white);
        bool insufficient_material(bool white);
        bool game_over();

        BoardStatus get_status();
        void print_status();

        int *get_material();
        int sum_material();
        int *get_influence(bool test_check);
        int sum_influence(bool test_check);

        void one_hot_encode(int *out);
        void one_hot_encode(float *out);

        float evaluate();
        float minimax(int depth, bool white, float alpha, float beta);
        void change_minimax(bool white, int depth);
    };

    enum OpeningType
    {
        SicilianDefense = CHESS_OPENING_E4,
        FrenchDefense = CHESS_OPENING_E4 + 1,
        RuyLopezOpening = CHESS_OPENING_E4 + 2,
        CaroKannDefense = CHESS_OPENING_E4 + 3,
        ItalianGame = CHESS_OPENING_E4 + 4,
        SicilianDefenseClosed = CHESS_OPENING_E4 + 5,
        ScandinavianDefense = CHESS_OPENING_E4 + 6,
        PircDefense = CHESS_OPENING_E4 + 7,
        SicilianDefenseAlapinVariation = CHESS_OPENING_E4 + 8,
        AlekhinesDefense = CHESS_OPENING_E4 + 9,
        KingsGambit = CHESS_OPENING_E4 + 10,
        ScotchGame = CHESS_OPENING_E4 + 11,
        ViennaGame = CHESS_OPENING_E4 + 12,
        QueensGambit = CHESS_OPENING_D4,
        SlavDefense = CHESS_OPENING_D4 + 1,
        KingsIndianDefense = CHESS_OPENING_D4 + 2,
        NimzoIndianDefense = CHESS_OPENING_D4 + 3,
        QueensIndianDefense = CHESS_OPENING_D4 + 4,
        CatalanOpening = CHESS_OPENING_D4 + 5,
        BogoIndianDefense = CHESS_OPENING_D4 + 6,
        GrunfeldDefense = CHESS_OPENING_D4 + 7,
        DutchDefense = CHESS_OPENING_D4 + 8,
        TrompowskyAttack = CHESS_OPENING_D4 + 9,
        BenkoGambit = CHESS_OPENING_D4 + 10,
        LondonSystem = CHESS_OPENING_D4 + 11,
        BenoniDefense = CHESS_OPENING_D4 + 12,
    };

    class Openings
    {
    public:
        static Board create(OpeningType typ);
        static Board create_rand_e4();
        static Board create_rand_d4();
    };
}
