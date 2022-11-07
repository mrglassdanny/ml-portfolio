#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <vector>

#define CHESS_BOARD_ROW_CNT 8
#define CHESS_BOARD_COL_CNT 8
#define CHESS_BOARD_LEN (CHESS_BOARD_COL_CNT * CHESS_BOARD_ROW_CNT)

#define CHESS_MAX_LEGAL_MOVE_CNT 64

#define CHESS_MAX_AN_MOVE_LEN 8
#define CHESS_MAX_GAME_MOVE_CNT 500

#define CHESS_INVALID_VALUE -1

#define CHESS_ONE_HOT_ENCODE_COMBINATION_CNT 6
#define CHESS_ONE_HOT_ENCODED_BOARD_LEN (CHESS_BOARD_LEN * CHESS_ONE_HOT_ENCODE_COMBINATION_CNT)

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

typedef enum PieceType
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
} PieceType;

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

class Board
{
private:
    int data_[CHESS_BOARD_LEN];
    float flt_data_[CHESS_BOARD_LEN];
    float influence_data_[CHESS_BOARD_LEN];
    int temp_piece_influence_idxs_[CHESS_MAX_LEGAL_MOVE_CNT];
    int temp_legal_move_idxs_[CHESS_MAX_LEGAL_MOVE_CNT];
    char temp_an_move_[CHESS_MAX_AN_MOVE_LEN];

public:
    Board();
    ~Board();

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

    int operator[](int);
    bool operator==(const Board &);
    bool operator!=(const Board &);

    void to_float();

    void reset();
    void copy(Board *src);
    void print(bool flip);

    int *get_legal_moves(int piece_idx, bool test_in_check_flg);
    Move get_random_move(bool white, Board *cmp_board);

    bool is_cell_under_attack(int idx, bool white);
    bool is_in_check(bool white);
    bool is_in_checkmate(bool white);
    bool is_in_stalemate(bool white);

    const char *translate_to_an_move(Move move);
    Move change(const char *an_move, bool white);
    Board simulate(Move move);

    int *get_piece_influence(int piece_idx);
    float *get_influence();
    void print_influence();
};

void one_hot_encode_board(int *board, int *out);
void one_hot_encode_board(int *board, float *out);
