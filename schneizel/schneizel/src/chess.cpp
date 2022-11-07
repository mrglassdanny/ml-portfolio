#include "chess.h"

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

int Board::operator[](int idx)
{
    return this->data_[idx];
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

void Board::print(bool flip)
{
    if (!flip)
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
    else
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
}

bool Board::is_cell_under_attack(int idx, bool white)
{
    bool under_attack_flg = false;

    int *board = this->data_;

    if (white)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (Piece::is_piece_black((PieceType)board[piece_idx]) && (PieceType)board[piece_idx] != PieceType::BlackKing)
            {
                int *legal_moves = this->get_legal_moves_for_piece(piece_idx, false);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

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
                int *legal_moves = this->get_legal_moves_for_piece(piece_idx, false);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

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

bool Board::is_in_check(bool white)
{
    bool in_check_flg = 0;

    int *board = this->data_;

    if (white)
    {
        for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
        {
            if (Piece::is_piece_black((PieceType)board[piece_idx]))
            {
                int *legal_moves = this->get_legal_moves_for_piece(piece_idx, false);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

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
                int *legal_moves = this->get_legal_moves_for_piece(piece_idx, false);

                for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
                {
                    if (legal_moves[mov_idx] == CHESS_INVALID_VALUE)
                    {
                        break;
                    }

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

bool Board::is_in_checkmate(bool white)
{
    bool in_checkmate_flg;

    int *board = this->data_;

    if (this->is_in_check(white))
    {
        in_checkmate_flg = true;

        if (white)
        {
            for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
            {
                if (Piece::is_piece_white((PieceType)board[piece_idx]))
                {
                    int *legal_moves = this->get_legal_moves_for_piece(piece_idx, true);

                    if (legal_moves[0] != CHESS_INVALID_VALUE)
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
                    int *legal_moves = this->get_legal_moves_for_piece(piece_idx, true);

                    if (legal_moves[0] != CHESS_INVALID_VALUE)
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

bool Board::is_in_stalemate(bool white)
{
    bool in_stalemate_flg;

    int *board = this->data_;

    if (!this->is_in_check(white))
    {
        in_stalemate_flg = true;

        if (white)
        {
            for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
            {
                if (Piece::is_piece_white((PieceType)board[piece_idx]))
                {
                    int *legal_moves = this->get_legal_moves_for_piece(piece_idx, true);

                    if (legal_moves[0] != CHESS_INVALID_VALUE)
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
                    int *legal_moves = this->get_legal_moves_for_piece(piece_idx, true);

                    if (legal_moves[0] != CHESS_INVALID_VALUE)
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

int *Board::get_legal_moves_for_piece(int piece_idx, bool test_in_check)
{
    int *out = this->temp_legal_move_idxs_;

    memset(out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
    int mov_ctr = 0;

    int *board = this->data_;

    PieceType piece = (PieceType)board[piece_idx];

    char col = Board::get_col_fr_idx(piece_idx);
    int row = Board::get_row_fr_idx(piece_idx);

    int adj_col = Board::get_adj_col_fr_idx(piece_idx);
    int adj_row = Board::get_adj_row_fr_idx(piece_idx);

    bool white = Piece::is_piece_white(piece);

    int test_idx;

    switch (piece)
    {
    case PieceType::WhitePawn:
        // TODO: au passant
        {
            test_idx = Board::get_idx_fr_colrow(col, row + 1);
            if (Board::is_row_valid(row + 1) && board[test_idx] == PieceType::Empty)
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row + 1);
            if (Board::is_adj_colrow_valid(adj_col - 1, adj_row + 1) && board[test_idx] != PieceType::Empty && !Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row + 1);
            if (Board::is_adj_colrow_valid(adj_col + 1, adj_row + 1) && board[test_idx] != PieceType::Empty && !Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
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
                        out[mov_ctr++] = test_idx;
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
                out[mov_ctr++] = test_idx;
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row - 1);
            if (Board::is_adj_colrow_valid(adj_col - 1, adj_row - 1) && board[test_idx] != PieceType::Empty && !Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row - 1);
            if (Board::is_adj_colrow_valid(adj_col + 1, adj_row - 1) && board[test_idx] != PieceType::Empty && !Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
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
                        out[mov_ctr++] = test_idx;
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
                out[mov_ctr++] = test_idx;
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 1, adj_row - 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 1, adj_row - 2);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 1, adj_row + 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row + 2);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 1, adj_row - 2))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 1, adj_row - 2);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 2, adj_row + 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 2, adj_row + 1);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (Board::is_adj_colrow_valid(adj_col + 2, adj_row - 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col + 2, adj_row - 1);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 2, adj_row + 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 2, adj_row + 1);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        if (Board::is_adj_colrow_valid(adj_col - 2, adj_row - 1))
        {
            test_idx = Board::get_idx_fr_adj_colrow(adj_col - 2, adj_row - 1);
            if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
            {
                out[mov_ctr++] = test_idx;
            }
        }
    }

    break;
    case PieceType::WhiteBishop:
    case PieceType::BlackBishop:
    {
        int ne = 0;
        int sw = 0;
        int se = 0;
        int nw = 0;
        for (int i = 1; i < 8; i++)
        {

            if (Board::is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
            {
                test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                if (board[test_idx] != PieceType::Empty)
                {
                    ne = 1;
                    if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (Board::is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
            {
                test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                if (board[test_idx] != PieceType::Empty)
                {
                    sw = 1;
                    if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (Board::is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
            {
                test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                if (board[test_idx] != PieceType::Empty)
                {
                    se = 1;
                    if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (Board::is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
            {
                test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                if (board[test_idx] != PieceType::Empty)
                {
                    nw = 1;
                    if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }
        }
    }

    break;
    case PieceType::WhiteRook:
    case PieceType::BlackRook:
    {
        int n = 0;
        int s = 0;
        int e = 0;
        int w = 0;
        for (int i = 1; i < 8; i++)
        {

            if (Board::is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
            {
                test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row);
                if (board[test_idx] != PieceType::Empty)
                {
                    e = 1;
                    if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (Board::is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
            {
                test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row);
                if (board[test_idx] != PieceType::Empty)
                {
                    w = 1;
                    if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (Board::is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
            {
                test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row + i);
                if (board[test_idx] != PieceType::Empty)
                {
                    n = 1;
                    if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (Board::is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
            {
                test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row - i);
                if (board[test_idx] != PieceType::Empty)
                {
                    s = 1;
                    if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }
        }
    }

    break;
    case PieceType::WhiteQueen:
    case PieceType::BlackQueen:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 8; i++)
            {

                if (Board::is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        ne = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        sw = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        se = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        nw = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }
        // n,s,e,w
        {
            int n = 0;
            int s = 0;
            int e = 0;
            int w = 0;
            for (int i = 1; i < 8; i++)
            {

                if (Board::is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        e = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        w = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        n = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        s = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case PieceType::WhiteKing:
    case PieceType::BlackKing:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 2; i++)
            {

                if (Board::is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        ne = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        sw = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        se = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        nw = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }
        // n,s,e,w
        {
            int n = 0;
            int s = 0;
            int e = 0;
            int w = 0;
            for (int i = 1; i < 2; i++)
            {

                if (Board::is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col + i, adj_row);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        e = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col - i, adj_row);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        w = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        n = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (Board::is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = Board::get_idx_fr_adj_colrow(adj_col, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        s = 1;
                        if (!Piece::is_piece_same_color(piece, (PieceType)board[test_idx]))
                        {
                            out[mov_ctr++] = test_idx;
                        }
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        // Castles.
        if (piece == PieceType::WhiteKing)
        {
            if (col == 'e' && row == 1)
            {
                // Queen side castle.
                if (board[Board::get_idx_fr_colrow('a', 1)] == PieceType::WhiteRook)
                {
                    if (board[Board::get_idx_fr_colrow('b', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('c', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('d', 1)] == PieceType::Empty &&
                        !Board::is_cell_under_attack(Board::get_idx_fr_colrow('b', 1), true) && !Board::is_cell_under_attack(Board::get_idx_fr_colrow('c', 1), true) && !Board::is_cell_under_attack(Board::get_idx_fr_colrow('d', 1), true))
                    {
                        out[mov_ctr++] = Board::get_idx_fr_colrow('c', 1);
                    }
                }

                // King side castle.
                if (board[Board::get_idx_fr_colrow('h', 1)] == PieceType::WhiteRook)
                {
                    if (board[Board::get_idx_fr_colrow('f', 1)] == PieceType::Empty && board[Board::get_idx_fr_colrow('g', 1)] == PieceType::Empty &&
                        !Board::is_cell_under_attack(Board::get_idx_fr_colrow('f', 1), true) && !Board::is_cell_under_attack(Board::get_idx_fr_colrow('g', 1), true))
                    {
                        out[mov_ctr++] = Board::get_idx_fr_colrow('g', 1);
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
                        !Board::is_cell_under_attack(Board::get_idx_fr_colrow('b', 8), false) && !Board::is_cell_under_attack(Board::get_idx_fr_colrow('c', 8), false) && !Board::is_cell_under_attack(Board::get_idx_fr_colrow('d', 8), false))
                    {
                        out[mov_ctr++] = Board::get_idx_fr_colrow('c', 8);
                    }
                }

                // King side castle.
                if (board[Board::get_idx_fr_colrow('h', 8)] == PieceType::BlackRook)
                {
                    if (board[Board::get_idx_fr_colrow('f', 8)] == PieceType::Empty && board[Board::get_idx_fr_colrow('g', 8)] == PieceType::Empty &&
                        !Board::is_cell_under_attack(Board::get_idx_fr_colrow('f', 8), false) && !Board::is_cell_under_attack(Board::get_idx_fr_colrow('g', 8), false))
                    {
                        out[mov_ctr++] = Board::get_idx_fr_colrow('g', 8);
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
        int check_out[CHESS_MAX_LEGAL_MOVE_CNT];
        memset(check_out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
        int check_mov_ctr = 0;
        for (int i = 0; i < mov_ctr; i++)
        {
            Board sim = this->simulate(Move{piece_idx, out[i]});
            if (!sim.is_in_check(white))
            {
                check_out[check_mov_ctr++] = out[i];
            }
        }

        memcpy(out, check_out, sizeof(int) * mov_ctr);
    }

    return out;
}

Move Board::get_random_move(bool white, Board *cmp)
{
    int piece_idxs[CHESS_BOARD_LEN];
    memset(piece_idxs, 0, sizeof(int) * CHESS_BOARD_LEN);

    // Get piece indexes.

    int piece_ctr = 0;
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        if (white)
        {
            if (Piece::is_piece_white((PieceType)this->data_[i]))
            {
                piece_idxs[piece_ctr++] = i;
            }
        }
        else
        {
            if (Piece::is_piece_black((PieceType)this->data_[i]))
            {
                piece_idxs[piece_ctr++] = i;
            }
        }
    }

    Move move{CHESS_INVALID_VALUE, CHESS_INVALID_VALUE};
    int max_try_cnt = 20;
    int try_ctr = 0;

    while (try_ctr < max_try_cnt)
    {
        int rand_piece_idx = rand() % piece_ctr;

        // Got our piece; now get moves.
        int legal_mov_ctr = 0;
        int *legal_moves = this->get_legal_moves_for_piece(piece_idxs[rand_piece_idx], true);
        for (int i = 0; i < CHESS_MAX_LEGAL_MOVE_CNT; i++)
        {
            if (legal_moves[i] == CHESS_INVALID_VALUE)
            {
                break;
            }
            else
            {
                legal_mov_ctr++;
            }
        }

        // If at least 1 move found, randomly make one and compare.
        if (legal_mov_ctr > 0)
        {
            int rand_legal_mov_idx = rand() % legal_mov_ctr;
            Board sim = this->simulate(Move{piece_idxs[rand_piece_idx], legal_moves[rand_legal_mov_idx]});

            // Make sure the same move was not made.
            if (*cmp != sim)
            {
                move.src_idx = piece_idxs[rand_piece_idx];
                move.dst_idx = legal_moves[rand_legal_mov_idx];
                break;
            }
        }

        try_ctr++;
    }

    return move;
}

const char *Board::translate_to_an_move(Move move)
{
    char *out = this->temp_an_move_;

    memset(out, 0, CHESS_MAX_AN_MOVE_LEN);

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
            memcpy(out, "O-O", 3);
            return out;
        }
        else if ((src_adj_col - dst_adj_col) == 2)
        {
            memcpy(out, "O-O-O", 5);
            return out;
        }
    }

    // Example format going forward: piece id|src col|src row|dst col|dst row|promo (or space)|promo piece id (or space)
    // ^always 7 chars

    int move_ctr = 0;

    out[move_ctr++] = piece_id;

    out[move_ctr++] = src_col;
    out[move_ctr++] = (char)(src_row + '0');

    out[move_ctr++] = dst_col;
    out[move_ctr++] = (char)(dst_row + '0');

    // Check for pawn promotion. If none, set last 2 chars to ' '.
    if ((piece == PieceType::WhitePawn && dst_row == 8) || (piece == PieceType::BlackPawn && dst_row == 1))
    {
        out[move_ctr++] = '=';
        out[move_ctr++] = 'Q';
    }
    else
    {
        out[move_ctr++] = ' ';
        out[move_ctr++] = ' ';
    }

    return out;
}

Move Board::change(const char *an_move, bool white)
{
    char mut_an_move[CHESS_MAX_AN_MOVE_LEN];
    memcpy(mut_an_move, an_move, CHESS_MAX_AN_MOVE_LEN);

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
    for (int i = CHESS_MAX_AN_MOVE_LEN - 1; i > 0; i--)
    {
        if (mut_an_move[i] == '+' || mut_an_move[i] == '#')
        {
            // Can safely just 0 out since we know '+'/'#' will be at the end of the move string.
            mut_an_move[i] = 0;
        }
    }

    // Remove 'x'.
    for (int i = 0; i < CHESS_MAX_AN_MOVE_LEN; i++)
    {
        if (mut_an_move[i] == 'x')
        {
            for (int j = i; j < CHESS_MAX_AN_MOVE_LEN - 1; j++)
            {
                mut_an_move[j] = mut_an_move[j + 1];
            }
            break;
        }
    }

    int mut_mov_len = strlen(mut_an_move);

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
        if (strcmp(mut_an_move, "O-O") == 0)
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

                int found = 0;
                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (board[i] == piece)
                    {
                        int *legal_moves = this->get_legal_moves_for_piece(i, true);
                        for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
                        {
                            if (legal_moves[j] == dst_idx)
                            {
                                board[dst_idx] = piece;
                                src_idx = i;
                                board[src_idx] = PieceType::Empty;
                                found = 1;
                                break;
                            }
                            else if (legal_moves[j] == CHESS_INVALID_VALUE)
                            {
                                break;
                            }
                        }
                    }
                    if (found == 1)
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

                int found = 0;
                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (board[i] == piece)
                    {
                        int *legal_moves = this->get_legal_moves_for_piece(i, true);
                        for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
                        {
                            if (legal_moves[j] == dst_idx)
                            {
                                src_idx = i;
                                board[src_idx] = PieceType::Empty;
                                found = 1;
                                break;
                            }
                            else if (legal_moves[j] == CHESS_INVALID_VALUE)
                            {
                                break;
                            }
                        }
                    }
                    if (found == 1)
                    {
                        break;
                    }
                }

                board[dst_idx] = promo_piece;
            }
        }
        break;
    case 5:
        if (strcmp(mut_an_move, "O-O-O") == 0)
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

    Move chess_move;
    chess_move.src_idx = src_idx;
    chess_move.dst_idx = dst_idx;
    return chess_move;
}

Move Board::change(Move move, bool white)
{
    const char *an_move = this->translate_to_an_move(move);
    return this->change(an_move, white);
}

Board Board::simulate(Move move)
{
    Board sim;
    sim.copy(this);

    const char *an_move = this->translate_to_an_move(move);

    PieceType piece = (PieceType)sim.data_[move.src_idx];

    if (Piece::is_piece_white(piece))
    {
        sim.change(an_move, true);
    }
    else
    {
        sim.change(an_move, false);
    }

    return sim;
}

std::vector<Board> Board::simulate_all_legal_moves(bool white)
{
    std::vector<Board> sims;

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        bool check_moves = false;
        if (white)
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
            int *legal_moves = this->get_legal_moves_for_piece(i, true);
            for (int j = 0; j < CHESS_MAX_LEGAL_MOVE_CNT; j++)
            {
                if (legal_moves[j] == CHESS_INVALID_VALUE)
                {
                    break;
                }

                Board sim = this->simulate(Move{i, legal_moves[j]});

                sims.push_back(sim);
            }
        }
    }

    return sims;
}

float *Board::get_float()
{
    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        this->flt_data_[i] = Piece::piece_to_float((PieceType)this->data_[i]);
    }

    return this->flt_data_;
}

void Board::print_float()
{
    this->get_float();

    // Print in a more viewable format(a8 at top left of screen).
    printf("   +---+---+---+---+---+---+---+---+");
    printf("\n");
    for (int i = CHESS_BOARD_ROW_CNT - 1; i >= 0; i--)
    {
        printf("%d  ", i + 1);
        printf("|");
        for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
        {
            int val = ceil(this->flt_data_[(i * CHESS_BOARD_COL_CNT) + j]);

            if (val < 0)
            {
                printf("%d |", val);
            }
            else if (val > 0)
            {
                printf(" %d |", val);
            }
            else
            {
                printf(" %d |", val);
            }
        }
        printf("\n");
        printf("   +---+---+---+---+---+---+---+---+");
        printf("\n");
    }

    printf("    ");
    for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
    {
        printf(" %c  ", Board::get_col_fr_adj_col(j));
    }

    printf("\n\n");
}

int *Board::get_piece_influence(int piece_idx)
{
    int *out = this->temp_piece_influence_idxs_;

    memset(out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
    int mov_ctr = 0;

    int *board = this->data_;

    PieceType piece = (PieceType)board[piece_idx];

    char col = get_col_fr_idx(piece_idx);
    int row = get_row_fr_idx(piece_idx);

    int adj_col = get_adj_col_fr_idx(piece_idx);
    int adj_row = get_adj_row_fr_idx(piece_idx);

    bool white = Piece::is_piece_white(piece);

    int test_idx;

    switch (piece)
    {
    case PieceType::WhitePawn:
        // TODO: au passant
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row + 1);
            if (is_adj_colrow_valid(adj_col - 1, adj_row + 1))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row + 1);
            if (is_adj_colrow_valid(adj_col + 1, adj_row + 1))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        break;
    case PieceType::BlackPawn:
        // TODO: au passant
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row - 1);
            if (is_adj_colrow_valid(adj_col - 1, adj_row - 1))
            {
                out[mov_ctr++] = test_idx;
            }

            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row - 1);
            if (is_adj_colrow_valid(adj_col + 1, adj_row - 1))
            {
                out[mov_ctr++] = test_idx;
            }
        }

        break;
    case PieceType::WhiteKnight:
    case PieceType::BlackKnight:
    {

        if (is_adj_colrow_valid(adj_col + 1, adj_row + 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row + 2);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col + 1, adj_row - 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 1, adj_row - 2);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col - 1, adj_row + 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row + 2);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col - 1, adj_row - 2))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 1, adj_row - 2);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col + 2, adj_row + 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 2, adj_row + 1);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col + 2, adj_row - 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col + 2, adj_row - 1);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col - 2, adj_row + 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 2, adj_row + 1);
            out[mov_ctr++] = test_idx;
        }

        if (is_adj_colrow_valid(adj_col - 2, adj_row - 1))
        {
            test_idx = get_idx_fr_adj_colrow(adj_col - 2, adj_row - 1);
            out[mov_ctr++] = test_idx;
        }
    }

    break;
    case PieceType::WhiteBishop:
    case PieceType::BlackBishop:
    {
        int ne = 0;
        int sw = 0;
        int se = 0;
        int nw = 0;
        for (int i = 1; i < 8; i++)
        {

            if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                if (board[test_idx] != PieceType::Empty)
                {
                    ne = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                if (board[test_idx] != PieceType::Empty)
                {
                    sw = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                if (board[test_idx] != PieceType::Empty)
                {
                    se = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                if (board[test_idx] != PieceType::Empty)
                {
                    nw = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }
        }
    }

    break;
    case PieceType::WhiteRook:
    case PieceType::BlackRook:
    {
        int n = 0;
        int s = 0;
        int e = 0;
        int w = 0;
        for (int i = 1; i < 8; i++)
        {

            if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                if (board[test_idx] != PieceType::Empty)
                {
                    e = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                if (board[test_idx] != PieceType::Empty)
                {
                    w = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                if (board[test_idx] != PieceType::Empty)
                {
                    n = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }

            if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
            {
                test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                if (board[test_idx] != PieceType::Empty)
                {
                    s = 1;
                    out[mov_ctr++] = test_idx;
                }
                else
                {
                    out[mov_ctr++] = test_idx;
                }
            }
        }
    }

    break;
    case PieceType::WhiteQueen:
    case PieceType::BlackQueen:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 8; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        ne = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        sw = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        se = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        nw = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }
        // n,s,e,w
        {
            int n = 0;
            int s = 0;
            int e = 0;
            int w = 0;
            for (int i = 1; i < 8; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        e = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        w = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        n = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        s = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    case PieceType::WhiteKing:
    case PieceType::BlackKing:
        // ne,sw,se,nw
        {
            int ne = 0;
            int sw = 0;
            int se = 0;
            int nw = 0;
            for (int i = 1; i < 2; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row + i) && ne == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        ne = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row - i) && sw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        sw = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col + i, adj_row - i) && se == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        se = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row + i) && nw == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        nw = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }
        // n,s,e,w
        {
            int n = 0;
            int s = 0;
            int e = 0;
            int w = 0;
            for (int i = 1; i < 2; i++)
            {

                if (is_adj_colrow_valid(adj_col + i, adj_row) && e == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col + i, adj_row);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        e = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col - i, adj_row) && w == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col - i, adj_row);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        w = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row + i) && n == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row + i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        n = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }

                if (is_adj_colrow_valid(adj_col, adj_row - i) && s == 0)
                {
                    test_idx = get_idx_fr_adj_colrow(adj_col, adj_row - i);
                    if (board[test_idx] != PieceType::Empty)
                    {
                        s = 1;
                        out[mov_ctr++] = test_idx;
                    }
                    else
                    {
                        out[mov_ctr++] = test_idx;
                    }
                }
            }
        }

        break;
    default: // Nothing...
        break;
    }

    // Test in check:
    {
        int check_out[CHESS_MAX_LEGAL_MOVE_CNT];
        memset(check_out, CHESS_INVALID_VALUE, sizeof(int) * CHESS_MAX_LEGAL_MOVE_CNT);
        int check_mov_ctr = 0;
        for (int i = 0; i < mov_ctr; i++)
        {
            Board sim = this->simulate(Move{piece_idx, out[i]});
            if (!sim.is_in_check(white))
            {
                check_out[check_mov_ctr++] = out[i];
            }
        }

        memcpy(out, check_out, sizeof(int) * mov_ctr);
    }

    return out;
}

float *Board::get_influence()
{
    float *out = this->influence_data_;
    memset(out, 0, sizeof(int) * CHESS_BOARD_LEN);

    for (int piece_idx = 0; piece_idx < CHESS_BOARD_LEN; piece_idx++)
    {
        PieceType piece = (PieceType)this->data_[piece_idx];

        if (piece != PieceType::Empty)
        {
            int *moves = this->get_piece_influence(piece_idx);

            for (int mov_idx = 0; mov_idx < CHESS_MAX_LEGAL_MOVE_CNT; mov_idx++)
            {
                int mov_dst_idx = moves[mov_idx];

                if (mov_dst_idx == CHESS_INVALID_VALUE)
                {
                    break;
                }

                if ((PieceType)this->data_[mov_dst_idx] == PieceType::Empty)
                {
                    if (Piece::is_piece_white(piece))
                    {
                        out[mov_dst_idx] += 0.5f;
                    }
                    else if (Piece::is_piece_black(piece))
                    {
                        out[mov_dst_idx] -= 0.5f;
                    }
                }
                else
                {
                    PieceType dst_piece = (PieceType)this->data_[mov_dst_idx];

                    if (Piece::is_piece_white(piece))
                    {
                        if (Piece::is_piece_same_color(piece, dst_piece))
                        {
                            out[mov_dst_idx] += 0.5f;
                        }
                        else
                        {
                            out[mov_dst_idx] += (abs(Piece::piece_to_float(dst_piece)) / 2.0f);
                        }
                    }
                    else if (Piece::is_piece_black(piece))
                    {
                        if (Piece::is_piece_same_color(piece, dst_piece))
                        {
                            out[mov_dst_idx] -= 0.5f;
                        }
                        else
                        {
                            out[mov_dst_idx] -= (abs(Piece::piece_to_float(dst_piece)) / 2.0f);
                        }
                    }
                }
            }
        }
    }

    return out;
}

void Board::print_influence()
{
    this->get_influence();

    // Print in a more viewable format(a8 at top left of screen).
    printf("   +---+---+---+---+---+---+---+---+");
    printf("\n");
    for (int i = CHESS_BOARD_ROW_CNT - 1; i >= 0; i--)
    {
        printf("%d  ", i + 1);
        printf("|");
        for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
        {
            int val = ceil(this->influence_data_[(i * CHESS_BOARD_COL_CNT) + j]);

            if (val < 0)
            {
                printf("%d |", val);
            }
            else if (val > 0)
            {
                printf(" %d |", val);
            }
            else
            {
                printf(" %d |", val);
            }
        }
        printf("\n");
        printf("   +---+---+---+---+---+---+---+---+");
        printf("\n");
    }

    printf("    ");
    for (int j = 0; j < CHESS_BOARD_COL_CNT; j++)
    {
        printf(" %c  ", Board::get_col_fr_adj_col(j));
    }

    printf("\n\n");
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
