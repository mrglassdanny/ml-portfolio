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

PieceType Piece::fr_char(char piece_id, bool white)
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

char Piece::fr_piece(PieceType piece)
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

bool Piece::is_white(PieceType piece)
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

bool Piece::is_black(PieceType piece)
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

bool Piece::is_same_color(PieceType a, PieceType b)
{
    if ((is_white(a) && is_white(b)) || (is_black(a) && is_black(b)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int Piece::to_int(PieceType typ)
{
    switch (typ)
    {
    case PieceType::WhitePawn:
        return 1;
    case PieceType::WhiteKnight:
        return 3;
    case PieceType::WhiteBishop:
        return 3;
    case PieceType::WhiteRook:
        return 5;
    case PieceType::WhiteQueen:
        return 9;
    case PieceType::WhiteKing:
        return 3;
    case PieceType::BlackPawn:
        return -1;
    case PieceType::BlackKnight:
        return -3;
    case PieceType::BlackBishop:
        return -3;
    case PieceType::BlackRook:
        return -5;
    case PieceType::BlackQueen:
        return -9;
    case PieceType::BlackKing:
        return -3;
    default:
        return 0;
    }
}

const char *Piece::to_str(PieceType typ)
{
    switch (typ)
    {
    case PieceType::WhitePawn:
        return "  P  |";
    case PieceType::BlackPawn:
        return "  p  |";
    case PieceType::WhiteKnight:
        return "  N  |";
    case PieceType::BlackKnight:
        return "  n  |";
    case PieceType::WhiteBishop:
        return "  B  |";
    case PieceType::BlackBishop:
        return "  b  |";
    case PieceType::WhiteRook:
        return "  R  |";
    case PieceType::BlackRook:
        return "  r  |";
    case PieceType::WhiteQueen:
        return "  Q  |";
    case PieceType::BlackQueen:
        return "  q  |";
    case PieceType::WhiteKing:
        return "  K  |";
    case PieceType::BlackKing:
        return "  k  |";
    default:
        return "     |";
    }
}

const char *Piece::to_pretty_str(PieceType typ)
{
    switch (typ)
    {
    case PieceType::WhitePawn:
        return " P ";
    case PieceType::BlackPawn:
        return " p ";
    case PieceType::WhiteKnight:
        return " N ";
    case PieceType::BlackKnight:
        return " n ";
    case PieceType::WhiteBishop:
        return " B ";
    case PieceType::BlackBishop:
        return " b ";
    case PieceType::WhiteRook:
        return " R ";
    case PieceType::BlackRook:
        return " r ";
    case PieceType::WhiteQueen:
        return " Q ";
    case PieceType::BlackQueen:
        return " q ";
    case PieceType::WhiteKing:
        return " K ";
    case PieceType::BlackKing:
        return " k ";
    default:
        return "   ";
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

bool Board::is_square_under_attack(int idx, bool white)
{
    bool under_attack_flg = false;

    int *board = this->data_;

    if (white)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (Piece::is_black((PieceType)board[piece_idx]) && (PieceType)board[piece_idx] != PieceType::BlackKing)
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
            if (Piece::is_white((PieceType)board[piece_idx]) && (PieceType)board[piece_idx] != PieceType::WhiteKing)
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
                if (!Piece::is_same_color(piece, (PieceType)board[test_idx]))
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
                if (!Piece::is_same_color(piece, (PieceType)board[test_idx]))
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
                if (!Piece::is_same_color(piece, (PieceType)board[test_idx]))
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
                if (!Piece::is_same_color(piece, (PieceType)board[test_idx]))
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
                if (!Piece::is_same_color(piece, (PieceType)board[test_idx]))
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
                if (!Piece::is_same_color(piece, (PieceType)board[test_idx]))
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
                if (!Piece::is_same_color(piece, (PieceType)board[test_idx]))
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
                if (!Piece::is_same_color(piece, (PieceType)board[test_idx]))
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
    memcpy(this->data_, CHESS_START_BOARD, sizeof(int) * (CHESS_BOARD_LEN));
}

void Board::copy(Board *src)
{
    memcpy(this->data_, src->data_, sizeof(int) * CHESS_BOARD_LEN);
}

void Board::print()
{
    this->print(BoardAnalysisType::PieceTypes);
}

void Board::print(BoardAnalysisType typ)
{
    const char *boundary = "\n   +-----+-----+-----+-----+-----+-----+-----+-----+\n";

    int *board;
    switch (typ)
    {
    case BoardAnalysisType::Material:
        board = this->get_material();
        break;
    case BoardAnalysisType::Influence:
        board = this->get_influence();
        break;
    case BoardAnalysisType::MaterialInfluenceMatrixMult:
        board = this->get_matinf_mtxmul();
        break;
    case BoardAnalysisType::MaterialInfluencePieceWise:
        board = this->get_matinf_piecewise();
        break;
    default:
        board = this->data_;
        break;
    }

    printf("%s", boundary);

    for (int i = CHESS_BOARD_ROW_CNT - 1; i >= 0; i--)
    {
        printf("%d  ", i + 1);
        printf("|");

        for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
        {
            int val = board[(i * CHESS_BOARD_COL_CNT) + j];
            switch (typ)
            {
            case BoardAnalysisType::Material:
            case BoardAnalysisType::Influence:
            case BoardAnalysisType::MaterialInfluenceMatrixMult:
            case BoardAnalysisType::MaterialInfluencePieceWise:
                if (val < 0)
                {
                    printf(" %d  |", val);
                }
                else if (val > 0)
                {
                    printf("  %d  |", val);
                }
                else
                {
                    printf("     |");
                }
                break;
            default:
                printf("%s", Piece::to_str((PieceType)val));
                break;
            }
        }

        printf("%s", boundary);
    }

    printf("    ");
    for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
    {
        printf("  %c   ", Board::get_col_fr_adj_col(j));
    }

    switch (typ)
    {
    case BoardAnalysisType::PieceTypes:
        this->print_status();
        printf("\n    Material: %d", this->sum_material());
        printf("\n    Influence: %d", this->sum_influence());
        break;
    case BoardAnalysisType::Material:
        printf("\n    Material: %d", this->sum_material());
        break;
    case BoardAnalysisType::Influence:
        printf("\n    Influence: %d", this->sum_influence());
        break;
    default:
        break;
    }

    printf("\n\n");
}

void Board::pretty_print()
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    int *board = this->data_;

    printf("\n");

    bool white_first = true;

    int foreground = 0;
    int background = 0;

    for (int i = CHESS_BOARD_ROW_CNT - 1; i >= 0; i--)
    {
        printf("%d  ", i + 1);
        printf("");

        for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
        {
            auto piece_typ = (PieceType)this->data_[i * CHESS_BOARD_COL_CNT + j];
            if (Piece::is_white(piece_typ))
            {
                foreground = 15;
            }
            else if (Piece::is_black(piece_typ))
            {
                foreground = 0;
            }
            else
            {
                foreground = 15;
            }

            if (j % 2 == 0)
            {
                if (white_first)
                {
                    background = 11;
                }
                else
                {
                    background = 8;
                }
            }
            else
            {
                if (white_first)
                {
                    background = 8;
                }
                else
                {
                    background = 11;
                }
            }

            FlushConsoleInputBuffer(hConsole);
            SetConsoleTextAttribute(hConsole, foreground + background * 16);

            printf("%s", Piece::to_pretty_str((PieceType)board[(i * CHESS_BOARD_COL_CNT) + j]));
        }

        white_first = !white_first;

        FlushConsoleInputBuffer(hConsole);
        SetConsoleTextAttribute(hConsole, 15);

        printf("\n");
    }

    FlushConsoleInputBuffer(hConsole);
    SetConsoleTextAttribute(hConsole, 15);

    printf("   ");
    for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
    {
        printf(" %c ", Board::get_col_fr_adj_col(j));
    }

    this->print_status();
    printf("\n    Material: %d", this->sum_material());
    printf("\n    Influence: %d", this->sum_influence());

    printf("\n\n");
}

std::vector<int> Board::get_piece_moves(int piece_idx, bool test_in_check)
{
    std::vector<int> out;

    int mov_ctr = 0;

    int *board = this->data_;

    PieceType piece_typ = (PieceType)board[piece_idx];
    bool white = Piece::is_white(piece_typ);

    char col = Board::get_col_fr_idx(piece_idx);
    int row = Board::get_row_fr_idx(piece_idx);

    int adj_col = Board::get_adj_col_fr_idx(piece_idx);
    int adj_row = Board::get_adj_row_fr_idx(piece_idx);

    int test_idx;

    switch (piece_typ)
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
            if (Board::is_adj_colrow_valid(adj_col - 1, adj_row + 1) && board[test_idx] != PieceType::Empty && !Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row + 1);
            if (Board::is_adj_colrow_valid(adj_col + 1, adj_row + 1) && board[test_idx] != PieceType::Empty && !Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
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
            if (Board::is_adj_colrow_valid(adj_col - 1, adj_row - 1) && board[test_idx] != PieceType::Empty && !Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row - 1);
            if (Board::is_adj_colrow_valid(adj_col + 1, adj_row - 1) && board[test_idx] != PieceType::Empty && !Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
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
            if (!Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 1, adj_row - 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row - 2);
            if (!Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 1, adj_row + 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row + 2);
            if (!Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 1, adj_row - 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row - 2);
            if (!Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 2, adj_row + 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 2, adj_row + 1);
            if (!Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 2, adj_row - 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 2, adj_row - 1);
            if (!Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 2, adj_row + 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 2, adj_row + 1);
            if (!Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 2, adj_row - 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 2, adj_row - 1);
            if (!Piece::is_same_color(piece_typ, (PieceType)board[test_idx]))
            {
                out.push_back(test_idx);
            }
        }
    }

    break;
    case PieceType::WhiteBishop:
    case PieceType::BlackBishop:
    {
        this->get_piece_diagonal_moves(piece_typ, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteRook:
    case PieceType::BlackRook:
    {
        this->get_piece_straight_moves(piece_typ, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteQueen:
    case PieceType::BlackQueen:
    {
        this->get_piece_diagonal_moves(piece_typ, adj_col, adj_row, 8, &out);
        this->get_piece_straight_moves(piece_typ, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteKing:
    case PieceType::BlackKing:
    {
        this->get_piece_diagonal_moves(piece_typ, adj_col, adj_row, 2, &out);
        this->get_piece_straight_moves(piece_typ, adj_col, adj_row, 2, &out);

        // Castles.
        if (piece_typ == PieceType::WhiteKing)
        {
            if (col == 'e' && row == 1)
            {
                // Queen side castle.
                if (board[Board::get_idx_fr_colrow('a', 1)] == PieceType::WhiteRook)
                {
                    if (board[Board::get_idx_fr_colrow('b', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('c', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('d', 1)] == PieceType::Empty &&
                        !Board::is_square_under_attack(Board::get_idx_fr_colrow('b', 1), white) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('c', 1), white) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('d', 1), white))
                    {
                        out.push_back(Board::get_idx_fr_colrow('c', 1));
                    }
                }

                // King side castle.
                if (board[Board::get_idx_fr_colrow('h', 1)] == PieceType::WhiteRook)
                {
                    if (board[Board::get_idx_fr_colrow('f', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('g', 1)] == PieceType::Empty &&
                        !Board::is_square_under_attack(Board::get_idx_fr_colrow('f', 1), white) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('g', 1), white))
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
                        !Board::is_square_under_attack(Board::get_idx_fr_colrow('b', 8), white) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('c', 8), white) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('d', 8), white))
                    {
                        out.push_back(Board::get_idx_fr_colrow('c', 8));
                    }
                }

                // King side castle.
                if (board[Board::get_idx_fr_colrow('h', 8)] == PieceType::BlackRook)
                {
                    if (board[Board::get_idx_fr_colrow('f', 8)] == PieceType::Empty && board[Board::get_idx_fr_colrow('g', 8)] == PieceType::Empty &&
                        !Board::is_square_under_attack(Board::get_idx_fr_colrow('f', 8), white) && !Board::is_square_under_attack(Board::get_idx_fr_colrow('g', 8), white))
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
            Board sim = this->simulate(Move{piece_idx, out[i]}, white);
            if (!sim.check(white))
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

    PieceType piece_typ = (PieceType)board[piece_idx];
    bool white = Piece::is_white(piece_typ);

    char col = Board::get_col_fr_idx(piece_idx);
    int row = Board::get_row_fr_idx(piece_idx);

    int adj_col = Board::get_adj_col_fr_idx(piece_idx);
    int adj_row = Board::get_adj_row_fr_idx(piece_idx);

    int test_idx;

    switch (piece_typ)
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
        this->get_piece_diagonal_influence(piece_typ, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteRook:
    case PieceType::BlackRook:
    {
        this->get_piece_straight_influence(piece_typ, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteQueen:
    case PieceType::BlackQueen:
    {
        this->get_piece_diagonal_influence(piece_typ, adj_col, adj_row, 8, &out);
        this->get_piece_straight_influence(piece_typ, adj_col, adj_row, 8, &out);
    }

    break;
    case PieceType::WhiteKing:
    case PieceType::BlackKing:
    {
        this->get_piece_diagonal_influence(piece_typ, adj_col, adj_row, 2, &out);
        this->get_piece_straight_influence(piece_typ, adj_col, adj_row, 2, &out);
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
            Board sim = this->simulate(Move{piece_idx, out[i]}, white);

            if (!sim.check(white))
            {
                upd_out.push_back(out[i]);
            }
        }
        out = upd_out;
    }

    return out;
}

Move Board::get_random_move(bool white)
{
    std::vector<int> piece_idxs;

    // Get piece indexes.

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        if (white)
        {
            if (Piece::is_white((PieceType)this->data_[i]))
            {
                piece_idxs.push_back(i);
            }
        }
        else
        {
            if (Piece::is_black((PieceType)this->data_[i]))
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
            Board sim = this->simulate(Move{piece_idxs[rand_piece_idx], legal_moves[rand_legal_mov_idx]}, white);

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
    char piece_id = Piece::fr_piece((PieceType)this->data_[move.src_idx]);
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

Move Board::change(std::string an_move, bool white)
{
    std::string mut_an_move = an_move;

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
    mut_an_move.erase(remove(mut_an_move.begin(), mut_an_move.end(), '+'), mut_an_move.end());
    mut_an_move.erase(remove(mut_an_move.begin(), mut_an_move.end(), '#'), mut_an_move.end());

    // Remove 'x'.
    mut_an_move.erase(remove(mut_an_move.begin(), mut_an_move.end(), 'x'), mut_an_move.end());

    switch (mut_an_move.size())
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
                piece = Piece::fr_char(mut_an_move[0], white);
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
            piece = Piece::fr_char(mut_an_move[0], white);

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
                piece = Piece::fr_char(piece_char, white);
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
                piece = Piece::fr_char(mut_an_move[0], white);
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
                    PieceType promo_piece = Piece::fr_char(promo_piece_char, white);

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
    case 7: // schneizel custom move format.
        piece = Piece::fr_char(mut_an_move[0], white);
        src_col = mut_an_move[1];
        src_row = get_row_fr_char(mut_an_move[2]);
        src_idx = get_idx_fr_colrow(src_col, src_row);
        dst_col = mut_an_move[3];
        dst_row = get_row_fr_char(mut_an_move[4]);
        dst_idx = get_idx_fr_colrow(dst_col, dst_row);

        if (mut_an_move[5] == '=')
        {
            PieceType promo_piece = Piece::fr_char(mut_an_move[6], white);
            board[dst_idx] = promo_piece;
            board[src_idx] = PieceType::Empty;
        }
        else
        {
            board[dst_idx] = piece;
            board[src_idx] = PieceType::Empty;
        }
        break;
    default: // Nothing..
        break;
    }

    Move chess_move{src_idx, dst_idx};
    return chess_move;
}

Move Board::change(Move move, bool white)
{
    auto an_move = this->convert_move_to_an_move(move);
    return this->change(an_move, white);
}

Move Board::change(bool white)
{
    return this->change(this->get_random_move(white), white);
}

Board Board::simulate(Move move, bool white)
{
    Board sim;
    sim.copy(this);

    sim.change(move, white);

    return sim;
}

std::vector<Board> Board::simulate_all_moves(bool white)
{
    std::vector<Board> sims;

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        bool check_moves = false;
        if (white)
        {
            if (Piece::is_white((PieceType)this->data_[i]))
            {
                check_moves = true;
            }
        }
        else
        {
            if (Piece::is_black((PieceType)this->data_[i]))
            {
                check_moves = true;
            }
        }

        if (check_moves)
        {
            std::vector<int> legal_moves = this->get_piece_moves(i, true);
            for (int j = 0; j < legal_moves.size(); j++)
            {
                Board sim = this->simulate(Move{i, legal_moves[j]}, white);
                sims.push_back(sim);
            }
        }
    }

    return sims;
}

bool Board::check(bool white)
{
    bool in_check_flg = false;
    int *board = this->data_;

    if (white)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (Piece::is_black((PieceType)board[piece_idx]))
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
            if (Piece::is_white((PieceType)board[piece_idx]))
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
                if (Piece::is_white((PieceType)board[piece_idx]))
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
                if (Piece::is_black((PieceType)board[piece_idx]))
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
                if (Piece::is_white((PieceType)board[piece_idx]))
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
                if (Piece::is_black((PieceType)board[piece_idx]))
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
    else
    {
        return BoardStatus::Normal;
    }
}

void Board::print_status()
{
    printf("\n    Status: ");

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
    default:
        break;
    }
}

int *Board::get_material()
{
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        this->material_data_[i] = Piece::to_int((PieceType)this->data_[i]);
    }

    return this->material_data_;
}

int Board::sum_material()
{
    this->get_material();

    int sum = 0;
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        sum += this->material_data_[i];
    }
    return sum;
}

int *Board::get_influence()
{
    memset(this->influence_data_, 0, sizeof(int) * CHESS_BOARD_LEN);

    for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
    {
        PieceType piece_typ = (PieceType)this->data_[piece_idx];

        if (piece_typ != PieceType::Empty)
        {
            auto piece_inf = this->get_piece_influence(piece_idx);

            for (int piece_inf_idx = 0; piece_inf_idx < piece_inf.size(); piece_inf_idx++)
            {
                int dst_idx = piece_inf[piece_inf_idx];

                int val = 0;

                if (Piece::is_white(piece_typ))
                {
                    val = 1;
                }
                else if (Piece::is_black(piece_typ))
                {
                    val = -1;
                }

                this->influence_data_[dst_idx] += val;
            }
        }
    }

    return this->influence_data_;
}

int Board::sum_influence()
{
    this->get_influence();

    int sum = 0;
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        sum += this->influence_data_[i];
    }
    return sum;
}

int *Board::get_matinf_mtxmul()
{
    int *mat = this->get_material();
    int *inf = this->get_influence();

    memset(this->matinf_mtxmul_data_, 0, sizeof(int) * CHESS_BOARD_LEN);

    for (int i = 0; i < CHESS_BOARD_ROW_CNT; i++)
    {
        for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
        {
            for (int k = 0; k < CHESS_BOARD_COL_CNT; k++)
            {
                this->matinf_mtxmul_data_[i * CHESS_BOARD_COL_CNT + j] +=
                    mat[i * CHESS_BOARD_COL_CNT + k] *
                    inf[k * CHESS_BOARD_COL_CNT + j];
            }
        }
    }

    return this->matinf_mtxmul_data_;
}

int *Board::get_matinf_piecewise()
{
    int *mat = this->get_material();
    int *inf = this->get_influence();

    memset(this->matinf_piecewise_data_, 0, sizeof(int) * CHESS_BOARD_LEN);

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        if (mat[i] > 0 && inf[i] < 0)
        {
            this->matinf_piecewise_data_[i] = -(abs(mat[i] * inf[i]));
        }
        else if (mat[i] < 0 && inf[i] > 0)
        {
            this->matinf_piecewise_data_[i] = abs(mat[i] * inf[i]);
        }
    }

    return this->matinf_piecewise_data_;
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

Board Openings::create(OpeningType typ)
{
    Board board;

    switch (typ)
    {
    case SicilianDefense:
        board.change("e4", true);
        board.change("c5", false);
        break;
    case FrenchDefense:
        board.change("e4", true);
        board.change("e6", false);
        break;
    case RuyLopezOpening:
        board.change("e4", true);
        board.change("e5", false);
        board.change("Nf3", true);
        board.change("Nc6", false);
        board.change("Bb5", true);
        break;
    case CaroKannDefense:
        board.change("e4", true);
        board.change("c6", false);
        break;
    case ItalianGame:
        board.change("e4", true);
        board.change("e5", false);
        board.change("Nf3", true);
        board.change("Nc6", false);
        board.change("Bc4", true);
        break;
    case SicilianDefenseClosed:
        board.change("e4", true);
        board.change("c5", false);
        board.change("Nc3", true);
        break;
    case ScandinavianDefense:
        board.change("e4", true);
        board.change("d5", false);
        break;
    case PircDefense:
        board.change("e4", true);
        board.change("d6", false);
        board.change("d4", true);
        board.change("Nf6", false);
        break;
    case SicilianDefenseAlapinVariation:
        board.change("e4", true);
        board.change("c5", false);
        board.change("c3", true);
        break;
    case AlekhinesDefense:
        board.change("e4", true);
        board.change("Nf6", false);
        break;
    case KingsGambit:
        board.change("e4", true);
        board.change("e5", false);
        board.change("f4", true);
        break;
    case ScotchGame:
        board.change("e4", true);
        board.change("e5", false);
        board.change("Nf3", true);
        board.change("Nc6", false);
        board.change("d4", true);
        break;
    case ViennaGame:
        board.change("e4", true);
        board.change("e5", false);
        board.change("Nc3", true);
        break;
    case QueensGambit:
        board.change("d4", true);
        board.change("d5", false);
        board.change("c4", true);
        break;
    case SlavDefense:
        board.change("d4", true);
        board.change("d5", false);
        board.change("c4", true);
        board.change("c6", false);
        break;
    case KingsIndianDefense:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("c4", true);
        board.change("g6", false);
        break;
    case NimzoIndianDefense:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("c4", true);
        board.change("e6", false);
        board.change("Nc3", true);
        board.change("Bb4", false);
        break;
    case QueensIndianDefense:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("c4", true);
        board.change("e6", false);
        board.change("Nf3", true);
        board.change("b6", false);
        break;
    case CatalanOpening:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("c4", true);
        board.change("e6", false);
        board.change("g3", true);
        break;
    case BogoIndianDefense:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("c4", true);
        board.change("e6", false);
        board.change("Nf3", true);
        board.change("Bb4+", false);
        break;
    case GrunfeldDefense:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("c4", true);
        board.change("g6", false);
        board.change("Nc3", true);
        board.change("d5", false);
        break;
    case DutchDefense:
        board.change("d4", true);
        board.change("f5", false);
        break;
    case TrompowskyAttack:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("Bg5", true);
        break;
    case BenkoGambit:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("c4", true);
        board.change("c5", false);
        board.change("d5", true);
        board.change("b5", false);
        break;
    case LondonSystem:
        board.change("d4", true);
        board.change("d5", false);
        board.change("Nf3", true);
        board.change("Nf6", false);
        board.change("Bf4", true);
        break;
    case BenoniDefense:
        board.change("d4", true);
        board.change("Nf6", false);
        board.change("c4", true);
        board.change("c5", false);
        board.change("d5", true);
        board.change("e6", false);
        board.change("Nc3", true);
        board.change("exd5", false);
        board.change("cxd5", true);
        board.change("d6", false);
        break;
    default:
        break;
    }

    return board;
}

Board Openings::create_rand_e4()
{
    OpeningType typ = (OpeningType)((rand() % CHESS_OPENING_E4_CNT) + CHESS_OPENING_E4);
    return Openings::create(typ);
}

Board Openings::create_rand_d4()
{
    OpeningType typ = (OpeningType)((rand() % CHESS_OPENING_D4_CNT) + CHESS_OPENING_D4);
    return Openings::create(typ);
}