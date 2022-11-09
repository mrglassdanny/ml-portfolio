#include "chess.h"

using namespace chess;

int CHESS_START_BOARD[CHESS_BOARD_LEN] = {
    PieceType::WhiteRook, PieceType::WhiteKnight, PieceType::WhiteBishop, PieceType::WhiteQueen, PieceType::WhiteKing, PieceType::WhiteBishop, PieceType::WhiteKnight, PieceType::WhiteRook,
    PieceType::WhitePawn, PieceType::WhitePawn, PieceType::WhitePawn, PieceType::WhitePawn, PieceType::WhitePawn, PieceType::WhitePawn, PieceType::WhitePawn, PieceType::WhitePawn,
    PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty,
    PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty,
    PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty,
    PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty, PieceType::Empty,
    PieceType::BlackPawn, PieceType::BlackPawn, PieceType::BlackPawn, PieceType::BlackPawn, PieceType::BlackPawn, PieceType::BlackPawn, PieceType::BlackPawn, PieceType::BlackPawn,
    PieceType::BlackRook, PieceType::BlackKnight, PieceType::BlackBishop, PieceType::BlackQueen, PieceType::BlackKing, PieceType::BlackBishop, PieceType::BlackKnight, PieceType::BlackRook};

PieceType Piece::get_piece_fr_char(char piece_id, bool white)
{
    switch (piece_id)
    {
    case 'N':
        if (white)
        {
            return PieceType::WhiteKnight;
        }
        else
        {
            return PieceType::BlackKnight;
        }
    case 'B':
        if (white)
        {
            return PieceType::WhiteBishop;
        }
        else
        {
            return PieceType::BlackBishop;
        }
    case 'R':
        if (white)
        {
            return PieceType::WhiteRook;
        }
        else
        {
            return PieceType::BlackRook;
        }
    case 'Q':
        if (white)
        {
            return PieceType::WhiteQueen;
        }
        else
        {
            return PieceType::BlackQueen;
        }
    case 'K':
        if (white)
        {
            return PieceType::WhiteKing;
        }
        else
        {
            return PieceType::BlackKing;
        }
    default:
        // Pawn will be 'P' (optional).
        if (white)
        {
            return PieceType::WhitePawn;
        }
        else
        {
            return PieceType::BlackPawn;
        }
    }
}

char Piece::get_char_fr_piece(PieceType piece)
{
    switch (piece)
    {
    case PieceType::WhiteKnight:
    case PieceType::BlackKnight:
        return 'N';
    case PieceType::WhiteBishop:
    case PieceType::BlackBishop:
        return 'B';
    case PieceType::WhiteRook:
    case PieceType::BlackRook:
        return 'R';
    case PieceType::WhiteQueen:
    case PieceType::BlackQueen:
        return 'Q';
    case PieceType::WhiteKing:
    case PieceType::BlackKing:
        return 'K';
    default:
        // Pawn.
        return 'P';
    }
}

bool Piece::is_piece_white(PieceType piece)
{
    if (piece > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Piece::is_piece_black(PieceType piece)
{
    if (piece < 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Piece::is_piece_same_color(PieceType a, PieceType b)
{
    if ((is_piece_white(a) && is_piece_white(b)) || (is_piece_black(a) && is_piece_black(b)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

float Piece::piece_to_float(PieceType piece)
{
    switch (piece)
    {
    case PieceType::WhitePawn:
        return 1.0f;
    case PieceType::WhiteKnight:
        return 3.2f;
    case PieceType::WhiteBishop:
        return 3.33f;
    case PieceType::WhiteRook:
        return 5.1f;
    case PieceType::WhiteQueen:
        return 8.8f;
    case PieceType::WhiteKing:
        return 3.0f;
    case PieceType::BlackPawn:
        return -1.0f;
    case PieceType::BlackKnight:
        return -3.2f;
    case PieceType::BlackBishop:
        return -3.33f;
    case PieceType::BlackRook:
        return -5.1f;
    case PieceType::BlackQueen:
        return -8.8f;
    case PieceType::BlackKing:
        return -3.0f;
    default: // ChessPiece::Empty space.
        return 0.0f;
    }
}

Board::Board()
{
    this->reset();
}

Board::~Board() {}

int Board::get_col_fr_adj_col(int adj_col)
{
    char col;

    switch (adj_col)
    {
    case 0:
        col = 'a';
        break;
    case 1:
        col = 'b';
        break;
    case 2:
        col = 'c';
        break;
    case 3:
        col = 'd';
        break;
    case 4:
        col = 'e';
        break;
    case 5:
        col = 'f';
        break;
    case 6:
        col = 'g';
        break;
    default:
        col = 'h';
        break;
    }

    return col;
}

int Board::get_adj_col_fr_col(char col)
{
    int adj_col = 0;
    switch (col)
    {
    case 'a':
        adj_col = 0;
        break;
    case 'b':
        adj_col = 1;
        break;
    case 'c':
        adj_col = 2;
        break;
    case 'd':
        adj_col = 3;
        break;
    case 'e':
        adj_col = 4;
        break;
    case 'f':
        adj_col = 5;
        break;
    case 'g':
        adj_col = 6;
        break;
    default:
        adj_col = 7;
        break;
    }

    return adj_col;
}

int Board::get_row_fr_char(char row)
{
    return (row - '0');
}

int Board::get_adj_row_fr_row(int row)
{
    return row - 1;
}

int Board::get_adj_col_fr_idx(int idx)
{
    return idx % CHESS_BOARD_COL_CNT;
}

int Board::get_adj_row_fr_idx(int idx)
{
    return idx / CHESS_BOARD_ROW_CNT;
}

char Board::get_col_fr_idx(int idx)
{
    int adj_col = get_adj_col_fr_idx(idx);
    switch (adj_col)
    {
    case 0:
        return 'a';
    case 1:
        return 'b';
    case 2:
        return 'c';
    case 3:
        return 'd';
    case 4:
        return 'e';
    case 5:
        return 'f';
    case 6:
        return 'g';
    default:
        return 'h';
    }
}

int Board::get_row_fr_idx(int idx)
{
    return get_adj_row_fr_idx(idx) + 1;
}

int Board::get_idx_fr_colrow(char col, int row)
{
    int adj_col = get_adj_col_fr_col(col);

    int adj_row = get_adj_row_fr_row(row);

    return (adj_row * CHESS_BOARD_COL_CNT) + adj_col;
}

int Board::get_idx_fr_adj_colrow(int adj_col, int adj_row)
{
    return (adj_row * CHESS_BOARD_COL_CNT) + adj_col;
}

bool Board::is_row_valid(int row)
{
    if (row >= 1 && row <= CHESS_BOARD_ROW_CNT)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Board::is_adj_colrow_valid(int adj_col, int adj_row)
{
    if (adj_col >= 0 && adj_col < CHESS_BOARD_COL_CNT &&
        adj_row >= 0 && adj_row < CHESS_BOARD_ROW_CNT)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Board::is_square_under_attack(int idx)
{
    bool under_attack_flg = false;

    int *board = this->data_;

    if (this->white_)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (Piece::is_piece_black((PieceType)board[piece_idx]) && (PieceType)board[piece_idx] != PieceType::BlackKing)
            {
                std::vector<int> legal_moves = this->get_piece_moves(piece_idx, false);

                for (int mov_idx = 0; mov_idx < legal_moves.size(); mov_idx++)
                {
                    if (legal_moves[mov_idx] == idx)
                    {
                        under_attack_flg = true;
                        break;
                    }
                }
            }

            if (under_attack_flg)
            {
                break;
            }
        }
    }
    else
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (Piece::is_piece_white((PieceType)board[piece_idx]) && (PieceType)board[piece_idx] != PieceType::WhiteKing)
            {
                std::vector<int> legal_moves = this->get_piece_moves(piece_idx, false);

                for (int mov_idx = 0; mov_idx < legal_moves.size(); mov_idx++)
                {
                    if (legal_moves[mov_idx] == idx)
                    {
                        under_attack_flg = true;
                        break;
                    }
                }
            }

            if (under_attack_flg)
            {
                break;
            }
        }
    }

    return under_attack_flg;
}

void Board::get_piece_straight_moves(PieceType piece, int adj_col, int adj_row, int cnt, std::vector<int> *out)
{
    int test_idx;

    int *board = this->data_;

    bool n = false;
    bool s = false;
    bool e = false;
    bool w = false;

    for (int i = 1; i < cnt; i++)
    {
        if (Board::is_adj_colrow_valid(adj_col + i, adj_row) && !e)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row);
            if (board[test_idx] != PieceType::Empty)
            {
                e = true;
                if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                {
                    out->push_back(test_idx);
                }
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - i, adj_row) && !w)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row);
            if (board[test_idx] != PieceType::Empty)
            {
                w = true;
                if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                {
                    out->push_back(test_idx);
                }
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col, adj_row + i) && !n)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row + i);
            if (board[test_idx] != PieceType::Empty)
            {
                n = true;
                if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                {
                    out->push_back(test_idx);
                }
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col, adj_row - i) && !s)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row - i);
            if (board[test_idx] != PieceType::Empty)
            {
                s = true;
                if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                {
                    out->push_back(test_idx);
                }
            }
            else
            {
                out->push_back(test_idx);
            }
        }
    }
}

void Board::get_piece_straight_influence(PieceType piece, int adj_col, int adj_row, int cnt, std::vector<int> *out)
{
    int test_idx;

    int *board = this->data_;

    bool n = false;
    bool s = false;
    bool e = false;
    bool w = false;

    for (int i = 1; i < cnt; i++)
    {
        if (Board::is_adj_colrow_valid(adj_col + i, adj_row) && !e)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row);
            if (board[test_idx] != PieceType::Empty)
            {
                e = true;
                out->push_back(test_idx);
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - i, adj_row) && !w)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row);
            if (board[test_idx] != PieceType::Empty)
            {
                w = true;
                out->push_back(test_idx);
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col, adj_row + i) && !n)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row + i);
            if (board[test_idx] != PieceType::Empty)
            {
                n = true;
                out->push_back(test_idx);
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col, adj_row - i) && !s)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row - i);
            if (board[test_idx] != PieceType::Empty)
            {
                s = true;
                out->push_back(test_idx);
            }
            else
            {
                out->push_back(test_idx);
            }
        }
    }
}

void Board::get_piece_diagonal_moves(PieceType piece, int adj_col, int adj_row, int cnt, std::vector<int> *out)
{
    int test_idx;

    int *board = this->data_;

    bool ne = false;
    bool sw = false;
    bool se = false;
    bool nw = false;

    for (int i = 1; i < cnt; i++)
    {
        if (Board::is_adj_colrow_valid(adj_col + i, adj_row + i) && !ne)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
            if (board[test_idx] != PieceType::Empty)
            {
                ne = true;
                if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                {
                    out->push_back(test_idx);
                }
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - i, adj_row - i) && !sw)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
            if (board[test_idx] != PieceType::Empty)
            {
                sw = true;
                if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                {
                    out->push_back(test_idx);
                }
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + i, adj_row - i) && !se)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
            if (board[test_idx] != PieceType::Empty)
            {
                se = true;
                if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                {
                    out->push_back(test_idx);
                }
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - i, adj_row + i) && !nw)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
            if (board[test_idx] != PieceType::Empty)
            {
                nw = true;
                if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                {
                    out->push_back(test_idx);
                }
            }
            else
            {
                out->push_back(test_idx);
            }
        }
    }
}

void Board::get_piece_diagonal_influence(PieceType piece, int adj_col, int adj_row, int cnt, std::vector<int> *out)
{
    int test_idx;

    int *board = this->data_;

    bool ne = false;
    bool sw = false;
    bool se = false;
    bool nw = false;

    for (int i = 1; i < cnt; i++)
    {
        if (Board::is_adj_colrow_valid(adj_col + i, adj_row + i) && !ne)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
            if (board[test_idx] != PieceType::Empty)
            {
                ne = true;
                out->push_back(test_idx);
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - i, adj_row - i) && !sw)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
            if (board[test_idx] != PieceType::Empty)
            {
                sw = true;
                out->push_back(test_idx);
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + i, adj_row - i) && !se)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
            if (board[test_idx] != PieceType::Empty)
            {
                se = true;
                out->push_back(test_idx);
            }
            else
            {
                out->push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - i, adj_row + i) && !nw)
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
            if (board[test_idx] != PieceType::Empty)
            {
                nw = true;
                out->push_back(test_idx);
            }
            else
            {
                out->push_back(test_idx);
            }
        }
    }
}

bool Board::operator==(const Board &other)
{
    return memcmp(this->data_, other.data_, sizeof(int) * CHESS_BOARD_LEN) == 0;
}

bool Board::operator!=(const Board &other)
{
    return !(*this == other);
}

void Board::reset()
{
    this->white_ = true;
    this->move_cnt_ = 0;
    memcpy(this->data_, CHESS_START_BOARD, sizeof(int) * (CHESS_BOARD_LEN));
}

void Board::copy(Board *src)
{
    this->white_ = src->white_;
    this->move_cnt_ = src->move_cnt_;
    memcpy(this->data_, src->data_, sizeof(int) * CHESS_BOARD_LEN);
}

void Board::print()
{
    // Print in a more viewable format(a8 at top left of screen).
    printf("   +---+---+---+---+---+---+---+---+");
    printf("\n");
    for (int i = CHESS_BOARD_ROW_CNT - 1; i >= 0; i--)
    {
        printf("%d  ", i + 1);
        printf("|");
        for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
        {
            switch ((PieceType)this->data_[(i * CHESS_BOARD_COL_CNT) + j])
            {
            case PieceType::WhitePawn:
                printf(" P |");
                break;
            case PieceType::BlackPawn:
                printf(" p |");
                break;
            case PieceType::WhiteKnight:
                printf(" N |");
                break;
            case PieceType::BlackKnight:
                printf(" n |");
                break;
            case PieceType::WhiteBishop:
                printf(" B |");
                break;
            case PieceType::BlackBishop:
                printf(" b |");
                break;
            case PieceType::WhiteRook:
                printf(" R |");
                break;
            case PieceType::BlackRook:
                printf(" r |");
                break;
            case PieceType::WhiteQueen:
                printf(" Q |");
                break;
            case PieceType::BlackQueen:
                printf(" q |");
                break;
            case PieceType::WhiteKing:
                printf(" K |");
                break;
            case PieceType::BlackKing:
                printf(" k |");
                break;
            default:
                printf("   |");
                break;
            }
        }
        printf("\n");
        printf("   +---+---+---+---+---+---+---+---+");
        printf("\n");
    }

    printf("    ");
    for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
    {
        printf(" %c  ", get_col_fr_adj_col(j));
    }

    printf("\n\n");
}

void Board::print_flipped()
{
    printf("   +---+---+---+---+---+---+---+---+");
    printf("\n");
    for (int i = 0; i < CHESS_BOARD_ROW_CNT; i++)
    {
        printf("%d  ", i + 1);
        printf("|");
        for (int j = CHESS_BOARD_COL_CNT - 1; j >= 0; j--)
        {
            switch ((PieceType)this->data_[(i * CHESS_BOARD_COL_CNT) + j])
            {
            case PieceType::WhitePawn:
                printf(" P |");
                break;
            case PieceType::BlackPawn:
                printf(" p |");
                break;
            case PieceType::WhiteKnight:
                printf(" N |");
                break;
            case PieceType::BlackKnight:
                printf(" n |");
                break;
            case PieceType::WhiteBishop:
                printf(" B |");
                break;
            case PieceType::BlackBishop:
                printf(" b |");
                break;
            case PieceType::WhiteRook:
                printf(" R |");
                break;
            case PieceType::BlackRook:
                printf(" r |");
                break;
            case PieceType::WhiteQueen:
                printf(" Q |");
                break;
            case PieceType::BlackQueen:
                printf(" q |");
                break;
            case PieceType::WhiteKing:
                printf(" K |");
                break;
            case PieceType::BlackKing:
                printf(" k |");
                break;
            default:
                printf("   |");
                break;
            }
        }
        printf("\n");
        printf("   +---+---+---+---+---+---+---+---+");
        printf("\n");
    }

    printf("    ");
    for (int j = CHESS_BOARD_COL_CNT - 1; j >= 0; j--)
    {
        printf(" %c  ", get_col_fr_adj_col(j));
    }

    printf("\n\n");
}

bool Board::is_white_turn()
{
    return this->white_;
}

std::vector<int> Board::get_piece_moves(int piece_idx, bool test_in_check)
{
    std::vector<int> out;

    int mov_ctr = 0;

    int *board = this->data_;

    PieceType piece = (PieceType)board[piece_idx];

    char col = Board::get_col_fr_idx(piece_idx);
    int row = Board::get_row_fr_idx(piece_idx);

    int adj_col = Board::get_adj_col_fr_idx(piece_idx);
    int adj_row = Board::get_adj_row_fr_idx(piece_idx);

    int test_idx;

    switch (piece)
    {
    case PieceType::WhitePawn:
        // TODO: au passant
        {
            test_idx = Board::get_idx_fr_colrow(col, row + 1);
            if (Board::is_row_valid(row + 1) && board[test_idx] == PieceType::Empty)
            {
                out.push_back(test_idx);
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row + 1);
            if (Board::is_adj_colrow_valid(adj_col - 1, adj_row + 1) && board[test_idx] != PieceType::Empty && !Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row + 1);
            if (Board::is_adj_colrow_valid(adj_col + 1, adj_row + 1) && board[test_idx] != PieceType::Empty && !Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }

            if (row == 2)
            {
                // Dont need to check if row adjustments are valid since we know that starting row is 2.
                test_idx = Board::get_idx_fr_colrow(col, row + 1);
                if (board[test_idx] == PieceType::Empty)
                {
                    test_idx = Board::get_idx_fr_colrow(col, row + 2);
                    if (board[test_idx] == PieceType::Empty)
                    {
                        out.push_back(test_idx);
                    }
                }
            }
        }

        break;
    case PieceType::BlackPawn:
        // TODO: au passant
        {
            test_idx = Board::get_idx_fr_colrow(col, row - 1);
            if (Board::is_row_valid(row - 1) && board[test_idx] == PieceType::Empty)
            {
                out.push_back(test_idx);
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row - 1);
            if (Board::is_adj_colrow_valid(adj_col - 1, adj_row - 1) && board[test_idx] != PieceType::Empty && !Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row - 1);
            if (Board::is_adj_colrow_valid(adj_col + 1, adj_row - 1) && board[test_idx] != PieceType::Empty && !Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }

            if (row == 7)
            {
                // Dont need to check if row adjustments are valid since we know that starting row is 7.
                test_idx = Board::get_idx_fr_colrow(col, row - 1);
                if (board[test_idx] == PieceType::Empty)
                {
                    test_idx = Board::get_idx_fr_colrow(col, row - 2);
                    if (board[test_idx] == PieceType::Empty)
                    {
                        out.push_back(test_idx);
                    }
                }
            }
        }

        break;
    case PieceType::WhiteKnight:
    case PieceType::BlackKnight:
    {
        if (Board::is_adj_colrow_valid(adj_col + 1, adj_row + 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row + 2);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 1, adj_row - 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row - 2);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 1, adj_row + 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row + 2);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 1, adj_row - 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row - 2);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 2, adj_row + 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 2, adj_row + 1);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 2, adj_row - 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 2, adj_row - 1);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 2, adj_row + 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 2, adj_row + 1);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 2, adj_row - 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 2, adj_row - 1);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }
    }

    break;
    case PieceType::WhiteBishop:
    case PieceType::BlackBishop:
    {
        this->get_piece_diagonal_moves(piece, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteRook:
    case PieceType::BlackRook:
    {
        this->get_piece_straight_moves(piece, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteQueen:
    case PieceType::BlackQueen:
    {
        this->get_piece_diagonal_moves(piece, adj_col, adj_row, 8, &out);
        this->get_piece_straight_moves(piece, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteKing:
    case PieceType::BlackKing:
    {
        this->get_piece_diagonal_moves(piece, adj_col, adj_row, 2, &out);
        this->get_piece_straight_moves(piece, adj_col, adj_row, 2, &out);

        // Castles.
        if (piece == PieceType::WhiteKing)
        {
            if (col == 'e' && row == 1)
            {
                // Queen side castle.
                if (board[Board::get_idx_fr_colrow('a', 1)] == PieceType::WhiteRook)
                {
                    if (board[Board::get_idx_fr_colrow('b', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('c', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('d', 1)] == PieceType::Empty &&
                        !Board::is_square_under_attack(Board::get_idx_fr_colrow('b', 1)) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('c', 1)) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('d', 1)))
                    {
                        out.push_back(Board::get_idx_fr_colrow('c', 1));
                    }
                }

                // King side castle.
                if (board[Board::get_idx_fr_colrow('h', 1)] == PieceType::WhiteRook)
                {
                    if (board[Board::get_idx_fr_colrow('f', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('g', 1)] == PieceType::Empty &&
                        !Board::is_square_under_attack(Board::get_idx_fr_colrow('f', 1)) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('g', 1)))
                    {
                        out.push_back(Board::get_idx_fr_colrow('g', 1));
                    }
                }
            }
        }
        else
        {
            if (col == 'e' && row == 8)
            {
                // Queen side castle.
                if (board[Board::get_idx_fr_colrow('a', 8)] == PieceType::BlackRook)
                {
                    if (board[Board::get_idx_fr_colrow('b', 8)] == PieceType::Empty && board[Board::get_idx_fr_colrow('c', 8)] == PieceType::Empty && board[Board::get_idx_fr_colrow('d', 8)] == PieceType::Empty &&
                        !Board::is_square_under_attack(Board::get_idx_fr_colrow('b', 8)) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('c', 8)) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('d', 8)))
                    {
                        out.push_back(Board::get_idx_fr_colrow('c', 8));
                    }
                }

                // King side castle.
                if (board[Board::get_idx_fr_colrow('h', 8)] == PieceType::BlackRook)
                {
                    if (board[Board::get_idx_fr_colrow('f', 8)] == PieceType::Empty && board[Board::get_idx_fr_colrow('g', 8)] == PieceType::Empty &&
                        !Board::is_square_under_attack(Board::get_idx_fr_colrow('f', 8)) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('g', 8)))
                    {
                        out.push_back(Board::get_idx_fr_colrow('g', 8));
                    }
                }
            }
        }
    }

    break;
    default: // Nothing...
        break;
    }

    if (test_in_check)
    {
        std::vector<int> upd_out;
        for (int i = 0; i < out.size(); i++)
        {
            Board sim = this->simulate(Move{piece_idx, out[i]});
            if (!sim.check(this->white_))
            {
                upd_out.push_back(out[i]);
            }
        }
        out = upd_out;
    }

    return out;
}

std::vector<int> Board::get_piece_influence(int piece_idx)
{
    std::vector<int> out;

    int *board = this->data_;

    PieceType piece = (PieceType)board[piece_idx];

    char col = Board::get_col_fr_idx(piece_idx);
    int row = Board::get_row_fr_idx(piece_idx);

    int adj_col = Board::get_adj_col_fr_idx(piece_idx);
    int adj_row = Board::get_adj_row_fr_idx(piece_idx);

    int test_idx;

    switch (piece)
    {
    case PieceType::WhitePawn:
        // TODO: au passant
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row + 1);
            if (Board::is_adj_colrow_valid(adj_col - 1, adj_row + 1))
            {
                out.push_back(test_idx);
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row + 1);
            if (Board::is_adj_colrow_valid(adj_col + 1, adj_row + 1))
            {
                out.push_back(test_idx);
            }
        }

        break;
    case PieceType::BlackPawn:
        // TODO: au passant
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row - 1);
            if (Board::is_adj_colrow_valid(adj_col - 1, adj_row - 1))
            {
                out.push_back(test_idx);
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row - 1);
            if (Board::is_adj_colrow_valid(adj_col + 1, adj_row - 1))
            {
                out.push_back(test_idx);
            }
        }

        break;
    case PieceType::WhiteKnight:
    case PieceType::BlackKnight:
    {
        if (Board::is_adj_colrow_valid(adj_col + 1, adj_row + 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row + 2);
            out.push_back(test_idx);
        }

        if (Board::is_adj_colrow_valid(adj_col + 1, adj_row - 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row - 2);
            out.push_back(test_idx);
        }

        if (Board::is_adj_colrow_valid(adj_col - 1, adj_row + 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row + 2);
            out.push_back(test_idx);
        }

        if (Board::is_adj_colrow_valid(adj_col - 1, adj_row - 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row - 2);
            out.push_back(test_idx);
        }

        if (Board::is_adj_colrow_valid(adj_col + 2, adj_row + 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 2, adj_row + 1);
            out.push_back(test_idx);
        }

        if (Board::is_adj_colrow_valid(adj_col + 2, adj_row - 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 2, adj_row - 1);
            out.push_back(test_idx);
        }

        if (Board::is_adj_colrow_valid(adj_col - 2, adj_row + 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 2, adj_row + 1);
            out.push_back(test_idx);
        }

        if (Board::is_adj_colrow_valid(adj_col - 2, adj_row - 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 2, adj_row - 1);
            out.push_back(test_idx);
        }
    }

    break;
    case PieceType::WhiteBishop:
    case PieceType::BlackBishop:
    {
        this->get_piece_diagonal_influence(piece, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteRook:
    case PieceType::BlackRook:
    {
        this->get_piece_straight_influence(piece, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteQueen:
    case PieceType::BlackQueen:
    {
        this->get_piece_diagonal_influence(piece, adj_col, adj_row, 8, &out);
        this->get_piece_straight_influence(piece, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteKing:
    case PieceType::BlackKing:
    {
        this->get_piece_diagonal_influence(piece, adj_col, adj_row, 2, &out);
        this->get_piece_straight_influence(piece, adj_col, adj_row, 2, &out);
    }

    break;
    default: // Nothing...
        break;
    }

    // Test in check:
    {
        std::vector<int> upd_out;
        for (int i = 0; i < out.size(); i++)
        {
            Board sim = this->simulate(Move{piece_idx, out[i]});
            if (!sim.check())
            {
                upd_out.push_back(out[i]);
            }
        }
        out = upd_out;
    }

    return out;
}

Move Board::get_random_move()
{
    std::vector<int> piece_idxs;

    // Get piece indexes.

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        if (this->white_)
        {
            if (Piece::is_piece_white((PieceType)this->data_[i]))
            {
                piece_idxs.push_back(i);
            }
        }
        else
        {
            if (Piece::is_piece_black((PieceType)this->data_[i]))
            {
                piece_idxs.push_back(i);
            }
        }
    }

    Move move{CHESS_INVALID_VALUE, CHESS_INVALID_VALUE};
    int max_try_cnt = 20;
    int try_ctr = 0;

    while (try_ctr < max_try_cnt)
    {
        int rand_piece_idx = rand() % piece_idxs.size();

        // Got our piece; now get moves.
        std::vector<int> legal_moves = this->get_piece_moves(piece_idxs[rand_piece_idx], true);

        // If at least 1 move found, randomly make one and compare.
        if (legal_moves.size() > 0)
        {
            int rand_legal_mov_idx = rand() % legal_moves.size();
            Board sim = this->simulate(Move{piece_idxs[rand_piece_idx], legal_moves[rand_legal_mov_idx]});

            move.src_idx = piece_idxs[rand_piece_idx];
            move.dst_idx = legal_moves[rand_legal_mov_idx];
        }

        try_ctr++;
    }

    return move;
}

std::string Board::convert_move_to_an_move(Move move)
{
    std::string out;

    PieceType piece = (PieceType)this->data_[move.src_idx];
    char piece_id = Piece::get_char_fr_piece((PieceType)this->data_[move.src_idx]);
    char src_col = Board::get_col_fr_idx(move.src_idx);
    int src_row = Board::get_row_fr_idx(move.src_idx);
    char dst_col = Board::get_col_fr_idx(move.dst_idx);
    int dst_row = Board::get_row_fr_idx(move.dst_idx);

    // Check for castle.
    if (piece == PieceType::WhiteKing || piece == PieceType::BlackKing)
    {
        int src_adj_col = Board::get_adj_col_fr_col(src_col);
        int src_adj_row = Board::Board::get_adj_row_fr_row(src_row);
        int dst_adj_col = get_adj_col_fr_col(dst_col);
        int dst_adj_row = Board::get_adj_row_fr_row(dst_row);

        if ((src_adj_col - dst_adj_col) == -2)
        {
            out = "O-O";
            return out;
        }
        else if ((src_adj_col - dst_adj_col) == 2)
        {
            out = "O-O-O";
            return out;
        }
    }

    // Example format going forward: piece id|src col|src row|dst col|dst row|promo (or space)|promo piece id (or space)
    // ^always 7 chars

    int move_ctr = 0;

    out += piece_id;

    out += src_col;
    out += (char)(src_row + '0');

    out += dst_col;
    out += (char)(dst_row + '0');

    // Check for pawn promotion. If none, set last 2 chars to ' '.
    if ((piece == PieceType::WhitePawn && dst_row == 8) || (piece == PieceType::BlackPawn && dst_row == 1))
    {
        out += '=';
        out += 'Q';
    }
    else
    {
        out += ' ';
        out += ' ';
    }

    return out;
}

Move Board::change(std::string an_move)
{
    std::string mut_an_move = an_move;

    bool white = this->white_;
    int *board = this->data_;

    int src_idx = CHESS_INVALID_VALUE;
    int dst_idx = CHESS_INVALID_VALUE;
    char src_col;
    char dst_col;
    int src_row;
    int dst_row;
    PieceType piece;
    char piece_char;

    // Trim '+'/'#'.
    for (int i = an_move.size() - 1; i > 0; i--)
    {
        if (mut_an_move[i] == '+' || mut_an_move[i] == '#')
        {
            // Can safely just 0 out since we know '+'/'#' will be at the end of the move string.
            mut_an_move[i] = 0;
        }
    }

    // Remove 'x'.
    for (int i = 0; i < an_move.size(); i++)
    {
        if (mut_an_move[i] == 'x')
        {
            for (int j = i; j < an_move.size() - 1; j++)
            {
                mut_an_move[j] = mut_an_move[j + 1];
            }
            break;
        }
    }

    int mut_mov_len = mut_an_move.size();

    switch (mut_mov_len)
    {
    case 2:
        // Pawn move.
        dst_row = get_row_fr_char(mut_an_move[1]);
        dst_idx = get_idx_fr_colrow(mut_an_move[0], dst_row);
        if (white)
        {
            int prev_idx = get_idx_fr_colrow(mut_an_move[0], dst_row - 1);
            int prev_idx_2 = get_idx_fr_colrow(mut_an_move[0], dst_row - 2);

            board[dst_idx] = PieceType::WhitePawn;

            if (board[prev_idx] == PieceType::WhitePawn)
            {
                board[prev_idx] = PieceType::Empty;
                src_idx = prev_idx;
            }
            else if (board[prev_idx_2] == PieceType::WhitePawn)
            {
                board[prev_idx_2] = PieceType::Empty;
                src_idx = prev_idx_2;
            }
        }
        else
        {
            int prev_idx = get_idx_fr_colrow(mut_an_move[0], dst_row + 1);
            int prev_idx_2 = get_idx_fr_colrow(mut_an_move[0], dst_row + 2);

            board[dst_idx] = PieceType::BlackPawn;

            if (board[prev_idx] == PieceType::BlackPawn)
            {
                board[prev_idx] = PieceType::Empty;
                src_idx = prev_idx;
            }
            else if (board[prev_idx_2] == PieceType::BlackPawn)
            {
                board[prev_idx_2] = PieceType::Empty;
                src_idx = prev_idx_2;
            }
        }
        break;
    case 3:
        if (mut_an_move.compare("O-O") == 0)
        {
            // King side castle.
            if (white)
            {
                int cur_rook_idx = get_idx_fr_colrow('h', 1);
                int cur_king_idx = get_idx_fr_colrow('e', 1);
                board[cur_rook_idx] = PieceType::Empty;
                board[cur_king_idx] = PieceType::Empty;

                int nxt_rook_idx = get_idx_fr_colrow('f', 1);
                int nxt_king_idx = get_idx_fr_colrow('g', 1);
                board[nxt_rook_idx] = PieceType::WhiteRook;
                board[nxt_king_idx] = PieceType::WhiteKing;

                src_idx = cur_king_idx;
                dst_idx = nxt_king_idx;
            }
            else
            {
                int cur_rook_idx = get_idx_fr_colrow('h', 8);
                int cur_king_idx = get_idx_fr_colrow('e', 8);
                board[cur_rook_idx] = PieceType::Empty;
                board[cur_king_idx] = PieceType::Empty;

                int nxt_rook_idx = get_idx_fr_colrow('f', 8);
                int nxt_king_idx = get_idx_fr_colrow('g', 8);
                board[nxt_rook_idx] = PieceType::BlackRook;
                board[nxt_king_idx] = PieceType::BlackKing;

                src_idx = cur_king_idx;
                dst_idx = nxt_king_idx;
            }
        }
        else
        {
            // Need to check if isupper since pawn move will not have piece id -- just the src col.
            if (isupper(mut_an_move[0]) == 1)
            {
                // Minor/major piece move.
                piece = Piece::get_piece_fr_char(mut_an_move[0], white);
                dst_col = mut_an_move[1];
                dst_row = get_row_fr_char(mut_an_move[2]);
                dst_idx = get_idx_fr_colrow(dst_col, dst_row);

                bool found = false;
                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (board[i] == piece)
                    {
                        std::vector<int> legal_moves = this->get_piece_moves(i, true);
                        for (int j = 0; j < legal_moves.size(); j++)
                        {
                            if (legal_moves[j] == dst_idx)
                            {
                                board[dst_idx] = piece;
                                src_idx = i;
                                board[src_idx] = PieceType::Empty;
                                found = true;
                                break;
                            }
                        }
                    }
                    if (found)
                    {
                        break;
                    }
                }
            }
            else
            {
                // Disambiguated pawn move.
                src_col = mut_an_move[0];
                dst_col = mut_an_move[1];
                dst_row = get_row_fr_char(mut_an_move[2]);
                dst_idx = get_idx_fr_colrow(dst_col, dst_row);

                if (white)
                {
                    piece = PieceType::WhitePawn;
                    src_idx = get_idx_fr_colrow(src_col, dst_row - 1);
                    board[src_idx] = PieceType::Empty;

                    // Check if en passant:
                    if ((PieceType)board[dst_idx] == PieceType::Empty)
                    {
                        int en_passant_pawn_idx = get_idx_fr_colrow(dst_col, dst_row - 1);
                        if ((PieceType)board[en_passant_pawn_idx] == PieceType::BlackPawn)
                        {
                            board[en_passant_pawn_idx] = PieceType::Empty;
                        }
                    }
                }
                else
                {
                    piece = PieceType::BlackPawn;
                    src_idx = get_idx_fr_colrow(src_col, dst_row + 1);
                    board[src_idx] = PieceType::Empty;

                    // Check if en passant:
                    if ((PieceType)board[dst_idx] == PieceType::Empty)
                    {
                        int en_passant_pawn_idx = get_idx_fr_colrow(dst_col, dst_row + 1);
                        if ((PieceType)board[en_passant_pawn_idx] == PieceType::WhitePawn)
                        {
                            board[en_passant_pawn_idx] = PieceType::Empty;
                        }
                    }
                }

                board[dst_idx] = piece;
            }
        }
        break;
    case 4:
        // Need to check if isupper since pawn move will not have piece id -- just the src col.
        if (isupper(mut_an_move[0]) == 1)
        {
            // Disambiguated minor/major piece move.
            dst_col = mut_an_move[2];
            dst_row = get_row_fr_char(mut_an_move[3]);
            piece = Piece::get_piece_fr_char(mut_an_move[0], white);

            if (isdigit(mut_an_move[1]))
            {
                src_row = get_row_fr_char(mut_an_move[1]);

                dst_idx = get_idx_fr_colrow(dst_col, dst_row);

                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (get_row_fr_idx(i) == src_row && board[i] == piece)
                    {
                        board[dst_idx] = piece;
                        src_idx = i;
                        board[src_idx] = PieceType::Empty;
                        break;
                    }
                }
            }
            else
            {
                src_col = mut_an_move[1];

                dst_idx = get_idx_fr_colrow(dst_col, dst_row);

                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (get_col_fr_idx(i) == src_col && board[i] == piece)
                    {
                        board[dst_idx] = piece;
                        src_idx = i;
                        board[src_idx] = PieceType::Empty;
                        break;
                    }
                }
            }
        }
        else
        {
            // Pawn promotion.
            if (mut_an_move[2] == '=')
            {
                dst_row = get_row_fr_char(mut_an_move[1]);
                dst_idx = get_idx_fr_colrow(mut_an_move[0], dst_row);
                piece_char = mut_an_move[3];
                piece = Piece::get_piece_fr_char(piece_char, white);
                PieceType promo_piece = piece;

                if (white)
                {
                    piece = PieceType::WhitePawn;
                }
                else
                {
                    piece = PieceType::BlackPawn;
                }

                bool found = false;
                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (board[i] == piece)
                    {
                        std::vector<int> legal_moves = this->get_piece_moves(i, true);
                        for (int j = 0; j < legal_moves.size(); j++)
                        {
                            if (legal_moves[j] == dst_idx)
                            {
                                src_idx = i;
                                board[src_idx] = PieceType::Empty;
                                found = true;
                                break;
                            }
                        }
                    }
                    if (found)
                    {
                        break;
                    }
                }

                board[dst_idx] = promo_piece;
            }
        }
        break;
    case 5:
        if (mut_an_move.compare("O-O-O") == 0)
        {
            // Queen side castle.
            if (white)
            {
                int cur_rook_idx = get_idx_fr_colrow('a', 1);
                int cur_king_idx = get_idx_fr_colrow('e', 1);
                board[cur_rook_idx] = PieceType::Empty;
                board[cur_king_idx] = PieceType::Empty;

                int nxt_rook_idx = get_idx_fr_colrow('d', 1);
                int nxt_king_idx = get_idx_fr_colrow('c', 1);
                board[nxt_rook_idx] = PieceType::WhiteRook;
                board[nxt_king_idx] = PieceType::WhiteKing;

                src_idx = cur_king_idx;
                dst_idx = nxt_king_idx;
            }
            else
            {
                int cur_rook_idx = get_idx_fr_colrow('a', 8);
                int cur_king_idx = get_idx_fr_colrow('e', 8);
                board[cur_rook_idx] = PieceType::Empty;
                board[cur_king_idx] = PieceType::Empty;

                int nxt_rook_idx = get_idx_fr_colrow('d', 8);
                int nxt_king_idx = get_idx_fr_colrow('c', 8);
                board[nxt_rook_idx] = PieceType::BlackRook;
                board[nxt_king_idx] = PieceType::BlackKing;

                src_idx = cur_king_idx;
                dst_idx = nxt_king_idx;
            }
        }
        else
        {
            // Need to check if isupper since pawn move will not have piece id -- just the src col.
            if (isupper(mut_an_move[0]) == 1)
            {
                // Disambiguated queen move.
                piece = Piece::get_piece_fr_char(mut_an_move[0], white);
                if (piece == PieceType::WhiteQueen || piece == PieceType::BlackQueen)
                {
                    src_col = mut_an_move[1];
                    src_row = get_row_fr_char(mut_an_move[2]);
                    dst_col = mut_an_move[3];
                    dst_row = get_row_fr_char(mut_an_move[4]);

                    src_idx = get_idx_fr_colrow(src_col, src_row);
                    dst_idx = get_idx_fr_colrow(dst_col, dst_row);

                    board[dst_idx] = piece;
                    board[src_idx] = PieceType::Empty;
                }
            }
            else
            {
                // Disambiguated pawn promotion.
                if (mut_an_move[3] == '=')
                {
                    src_col = mut_an_move[0];
                    dst_col = mut_an_move[1];
                    dst_row = get_row_fr_char(mut_an_move[2]);
                    dst_idx = get_idx_fr_colrow(dst_col, dst_row);
                    char promo_piece_char = mut_an_move[4];
                    PieceType promo_piece = Piece::get_piece_fr_char(promo_piece_char, white);

                    if (white)
                    {
                        src_row = dst_row - 1;
                        piece = PieceType::WhitePawn;
                    }
                    else
                    {
                        src_row = dst_row + 1;
                        piece = PieceType::BlackPawn;
                    }

                    src_idx = get_idx_fr_colrow(src_col, src_row);
                    board[dst_idx] = promo_piece;
                    board[src_idx] = PieceType::Empty;
                }
            }
        }
        break;
    case 7: // 7 is chess-zero custom move format.
        if (mut_mov_len == 7)
        {
            piece = Piece::get_piece_fr_char(mut_an_move[0], white);
            src_col = mut_an_move[1];
            src_row = get_row_fr_char(mut_an_move[2]);
            src_idx = get_idx_fr_colrow(src_col, src_row);
            dst_col = mut_an_move[3];
            dst_row = get_row_fr_char(mut_an_move[4]);
            dst_idx = get_idx_fr_colrow(dst_col, dst_row);

            if (mut_an_move[5] == '=')
            {
                PieceType promo_piece = Piece::get_piece_fr_char(mut_an_move[6], white);
                board[dst_idx] = promo_piece;
                board[src_idx] = PieceType::Empty;
            }
            else
            {
                board[dst_idx] = piece;
                board[src_idx] = PieceType::Empty;
            }
        }
        break;
    default: // Nothing..
        break;
    }

    this->white_ = !this->white_;
    this->move_cnt_++;

    Move chess_move{src_idx, dst_idx};
    return chess_move;
}

Move Board::change(Move move)
{
    auto an_move = this->convert_move_to_an_move(move);
    return this->change(an_move);
}

Move Board::change()
{
    return this->change(this->get_random_move());
}

Board Board::simulate(Move move)
{
    Board sim;
    sim.copy(this);

    sim.change(this->convert_move_to_an_move(move));

    return sim;
}

std::vector<Board> Board::simulate_all_moves()
{
    std::vector<Board> sims;

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        bool check_moves = false;
        if (this->white_)
        {
            if (Piece::is_piece_white((PieceType)this->data_[i]))
            {
                check_moves = true;
            }
        }
        else
        {
            if (Piece::is_piece_black((PieceType)this->data_[i]))
            {
                check_moves = true;
            }
        }

        if (check_moves)
        {
            std::vector<int> legal_moves = this->get_piece_moves(i, true);
            for (int j = 0; j < legal_moves.size(); j++)
            {
                Board sim = this->simulate(Move{i, legal_moves[j]});

                sims.push_back(sim);
            }
        }
    }

    return sims;
}

bool Board::check()
{
    return this->check(this->white_);
}

bool Board::check(bool white)
{
    bool in_check_flg = false;
    int *board = this->data_;

    if (white)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (Piece::is_piece_black((PieceType)board[piece_idx]))
            {
                std::vector<int> legal_moves = this->get_piece_moves(piece_idx, false);

                for (int mov_idx = 0; mov_idx < legal_moves.size(); mov_idx++)
                {
                    if ((PieceType)board[legal_moves[mov_idx]] == PieceType::WhiteKing)
                    {
                        in_check_flg = true;
                        break;
                    }
                }
            }

            if (in_check_flg)
            {
                break;
            }
        }
    }
    else
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (Piece::is_piece_white((PieceType)board[piece_idx]))
            {
                std::vector<int> legal_moves = this->get_piece_moves(piece_idx, false);

                for (int mov_idx = 0; mov_idx < legal_moves.size(); mov_idx++)
                {
                    if ((PieceType)board[legal_moves[mov_idx]] == PieceType::BlackKing)
                    {
                        in_check_flg = true;
                        break;
                    }
                }
            }

            if (in_check_flg)
            {
                break;
            }
        }
    }

    return in_check_flg;
}

bool Board::checkmate()
{
    return this->checkmate(this->white_);
}

bool Board::checkmate(bool white)
{
    bool in_checkmate_flg;

    int *board = this->data_;

    if (this->check(white))
    {
        in_checkmate_flg = true;

        if (white)
        {
            for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
            {
                if (Piece::is_piece_white((PieceType)board[piece_idx]))
                {
                    std::vector<int> legal_moves = this->get_piece_moves(piece_idx, true);

                    if (legal_moves.size() > 0)
                    {
                        in_checkmate_flg = false;
                        break;
                    }
                }
            }
        }
        else
        {
            for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
            {
                if (Piece::is_piece_black((PieceType)board[piece_idx]))
                {
                    std::vector<int> legal_moves = this->get_piece_moves(piece_idx, true);

                    if (legal_moves.size() > 0)
                    {
                        in_checkmate_flg = false;
                        break;
                    }
                }
            }
        }
    }
    else
    {
        in_checkmate_flg = false;
    }

    return in_checkmate_flg;
}

bool Board::stalemate()
{
    return this->stalemate(this->white_);
}

bool Board::stalemate(bool white)
{
    bool in_stalemate_flg;

    int *board = this->data_;

    if (!this->check(white))
    {
        in_stalemate_flg = true;

        if (white)
        {
            for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
            {
                if (Piece::is_piece_white((PieceType)board[piece_idx]))
                {
                    std::vector<int> legal_moves = this->get_piece_moves(piece_idx, true);

                    if (legal_moves.size() > 0)
                    {
                        in_stalemate_flg = false;
                        break;
                    }
                }
            }
        }
        else
        {
            for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
            {
                if (Piece::is_piece_black((PieceType)board[piece_idx]))
                {
                    std::vector<int> legal_moves = this->get_piece_moves(piece_idx, true);

                    if (legal_moves.size() > 0)
                    {
                        in_stalemate_flg = false;
                        break;
                    }
                }
            }
        }
    }
    else
    {
        in_stalemate_flg = false;
    }

    return in_stalemate_flg;
}

bool Board::insufficient_material()
{
    return this->insufficient_material(this->white_);
}

bool Board::insufficient_material(bool white)
{
    bool pawn_found = false;
    int knight_cnt = 0;
    int bishop_cnt = 0;
    bool rook_or_queen_found = false;

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        PieceType typ = (PieceType)this->data_[i];

        switch (typ)
        {
        case PieceType::WhitePawn:
            if (white)
            {
                pawn_found = true;
            }
            break;
        case PieceType::BlackPawn:
            if (!white)
            {
                pawn_found = true;
            }
            break;
        case PieceType::WhiteKnight:
            if (white)
            {
                knight_cnt++;
            }
            break;
        case PieceType::BlackKnight:
            if (!white)
            {
                knight_cnt++;
            }
            break;
        case PieceType::WhiteBishop:
            if (white)
            {
                bishop_cnt++;
            }
            break;
        case PieceType::BlackBishop:
            if (!white)
            {
                bishop_cnt++;
            }
            break;
        case PieceType::WhiteRook:
        case PieceType::WhiteQueen:
            if (white)
            {
                rook_or_queen_found = true;
                return false;
            }
        case PieceType::BlackRook:
        case PieceType::BlackQueen:
            if (!white)
            {
                rook_or_queen_found = true;
                return false;
            }
        default:
            break;
        }
    }

    if (knight_cnt < 2 && bishop_cnt == 0 &&
        !rook_or_queen_found && !pawn_found)
    {
        return true;
    }
    else if (knight_cnt == 0 && bishop_cnt < 2 &&
             !rook_or_queen_found && !pawn_found)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Board::game_over()
{
    BoardStatus sts = this->get_status();
    if (sts == BoardStatus::Normal || sts == BoardStatus::WhiteInCheck || sts == BoardStatus::BlackInCheck)
    {
        return false;
    }
    else
    {
        return true;
    }
}

BoardStatus Board::get_status()
{
    if (this->check(true))
    {
        return BoardStatus::WhiteInCheck;
    }
    else if (this->check(false))
    {
        return BoardStatus::BlackInCheck;
    }
    else if (this->checkmate(true))
    {
        return BoardStatus::WhiteInCheckmate;
    }
    else if (this->checkmate(false))
    {
        return BoardStatus::BlackInCheckmate;
    }
    else if (this->stalemate(true))
    {
        return BoardStatus::WhiteInStalemate;
    }
    else if (this->stalemate(false))
    {
        return BoardStatus::BlackInStalemate;
    }
    else if (this->insufficient_material(true))
    {
        return BoardStatus::WhiteInsufficientMaterial;
    }
    else if (this->insufficient_material(false))
    {
        return BoardStatus::BlackInsufficientMaterial;
    }
    else if (this->move_cnt_ >= CHESS_MOVE_LIMIT)
    {
        return BoardStatus::MoveLimitExceeded;
    }
    else
    {
        return BoardStatus::Normal;
    }
}

void Board::print_status()
{
    printf("Status: ");

    switch (this->get_status())
    {
    case BoardStatus::Normal:
        printf("Normal");
        break;
    case BoardStatus::WhiteInCheck:
        printf("WhiteInCheck");
        break;
    case BoardStatus::BlackInCheck:
        printf("BlackInCheck");
        break;
    case BoardStatus::WhiteInCheckmate:
        printf("WhiteInCheckmate");
        break;
    case BoardStatus::BlackInCheckmate:
        printf("BlackInCheckmate");
        break;
    case BoardStatus::WhiteInStalemate:
        printf("WhiteInStalemate");
        break;
    case BoardStatus::BlackInStalemate:
        printf("BlackInStalemate");
        break;
    case BoardStatus::WhiteInsufficientMaterial:
        printf("WhiteInsufficientMaterial");
        break;
    case BoardStatus::BlackInsufficientMaterial:
        printf("BlackInsufficientMaterial");
        break;
    case BoardStatus::MoveLimitExceeded:
        printf("MoveLimitExceeded");
        break;
    default:
        break;
    }

    printf("\tMove count: %d\n", this->move_cnt_);
}

void Board::one_hot_encode(int *out)
{
    memset(out, 0, sizeof(int) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        switch ((PieceType)this->data_[i])
        {
        case PieceType::WhitePawn:
            out[i] = 1;
            break;
        case PieceType::WhiteKnight:
            out[i + CHESS_BOARD_LEN] = 1;
            break;
        case PieceType::WhiteBishop:
            out[i + (CHESS_BOARD_LEN * 2)] = 1;
            break;
        case PieceType::WhiteRook:
            out[i + (CHESS_BOARD_LEN * 3)] = 1;
            break;
        case PieceType::WhiteQueen:
            out[i + (CHESS_BOARD_LEN * 4)] = 1;
            break;
        case PieceType::WhiteKing:
            out[i + (CHESS_BOARD_LEN * 5)] = 1;
            break;
        case PieceType::BlackPawn:
            out[i] = -1;
            break;
        case PieceType::BlackKnight:
            out[i + (CHESS_BOARD_LEN)] = -1;
            break;
        case PieceType::BlackBishop:
            out[i + (CHESS_BOARD_LEN * 2)] = -1;
            break;
        case PieceType::BlackRook:
            out[i + (CHESS_BOARD_LEN * 3)] = -1;
            break;
        case PieceType::BlackQueen:
            out[i + (CHESS_BOARD_LEN * 4)] = -1;
            break;
        case PieceType::BlackKing:
            out[i + (CHESS_BOARD_LEN * 5)] = -1;
            break;
        default: // ChessPiece::Empty:
            break;
        }
    }
}

void Board::one_hot_encode(float *out)
{
    memset(out, 0, sizeof(float) * CHESS_ONE_HOT_ENCODED_BOARD_LEN);

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        switch ((PieceType)this->data_[i])
        {
        case PieceType::WhitePawn:
            out[i] = 1.0f;
            break;
        case PieceType::WhiteKnight:
            out[i + CHESS_BOARD_LEN] = 1.0f;
            break;
        case PieceType::WhiteBishop:
            out[i + (CHESS_BOARD_LEN * 2)] = 1.0f;
            break;
        case PieceType::WhiteRook:
            out[i + (CHESS_BOARD_LEN * 3)] = 1.0f;
            break;
        case PieceType::WhiteQueen:
            out[i + (CHESS_BOARD_LEN * 4)] = 1.0f;
            break;
        case PieceType::WhiteKing:
            out[i + (CHESS_BOARD_LEN * 5)] = 1.0f;
            break;
        case PieceType::BlackPawn:
            out[i] = -1.0f;
            break;
        case PieceType::BlackKnight:
            out[i + (CHESS_BOARD_LEN)] = -1.0f;
            break;
        case PieceType::BlackBishop:
            out[i + (CHESS_BOARD_LEN * 2)] = -1.0f;
            break;
        case PieceType::BlackRook:
            out[i + (CHESS_BOARD_LEN * 3)] = -1.0f;
            break;
        case PieceType::BlackQueen:
            out[i + (CHESS_BOARD_LEN * 4)] = -1.0f;
            break;
        case PieceType::BlackKing:
            out[i + (CHESS_BOARD_LEN * 5)] = -1.0f;
            break;
        default: // ChessPiece::Empty:
            break;
        }
    }
}
