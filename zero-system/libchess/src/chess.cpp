#include "chess.h"

using namespace chess;

char CHESS_BOARD_START_STATE[CHESS_BOARD_LEN] =
    {
        CHESS_WR, CHESS_WN, CHESS_WB, CHESS_WQ, CHESS_WK, CHESS_WB, CHESS_WN, CHESS_WR,
        CHESS_WP, CHESS_WP, CHESS_WP, CHESS_WP, CHESS_WP, CHESS_WP, CHESS_WP, CHESS_WP,
        CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT,
        CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT,
        CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT,
        CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT, CHESS_MT,
        CHESS_BP, CHESS_BP, CHESS_BP, CHESS_BP, CHESS_BP, CHESS_BP, CHESS_BP, CHESS_BP,
        CHESS_BR, CHESS_BN, CHESS_BB, CHESS_BQ, CHESS_BK, CHESS_BB, CHESS_BN, CHESS_BR};

bool Move::is_valid(Move *move)
{
    return move->src_square != CHESS_INVALID_SQUARE && move->dst_square != CHESS_INVALID_SQUARE;
}

bool Piece::is_white(char piece)
{
    switch (piece)
    {
    case CHESS_WP:
    case CHESS_WN:
    case CHESS_WB:
    case CHESS_WR:
    case CHESS_WQ:
    case CHESS_WK:
        return true;
    default:
        return false;
    }
}

bool Piece::is_black(char piece)
{
    switch (piece)
    {
    case CHESS_BP:
    case CHESS_BN:
    case CHESS_BB:
    case CHESS_BR:
    case CHESS_BQ:
    case CHESS_BK:
        return true;
    default:
        return false;
    }
}

bool Piece::is_same_color(char piece_a, char piece_b)
{
    if ((Piece::is_white(piece_a) && Piece::is_white(piece_b)) ||
        (Piece::is_black(piece_a) && Piece::is_black(piece_b)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

const char *Piece::to_str(char piece)
{
    switch (piece)
    {
    case CHESS_WP:
        return " P ";
    case CHESS_BP:
        return " p ";
    case CHESS_WN:
        return " N ";
    case CHESS_BN:
        return " n ";
    case CHESS_WB:
        return " B ";
    case CHESS_BB:
        return " b ";
    case CHESS_WR:
        return " R ";
    case CHESS_BR:
        return " r ";
    case CHESS_WQ:
        return " Q ";
    case CHESS_BQ:
        return " q ";
    case CHESS_WK:
        return " K ";
    case CHESS_BK:
        return " k ";
    default:
        return "   ";
    }
}

int Piece::get_value(char piece)
{
    switch (piece)
    {
    case CHESS_WP:
        return 1;
    case CHESS_BP:
        return -1;
    case CHESS_WN:
        return 3;
    case CHESS_BN:
        return -3;
    case CHESS_WB:
        return 3;
    case CHESS_BB:
        return -3;
    case CHESS_WR:
        return 5;
    case CHESS_BR:
        return -5;
    case CHESS_WQ:
        return 9;
    case CHESS_BQ:
        return -9;
    case CHESS_WK:
        return 2;
    case CHESS_BK:
        return -2;
    default:
        return 0;
    }
}

char Piece::get_pgn_piece(char piece)
{
    switch (piece)
    {
    case CHESS_WN:
    case CHESS_BN:
        return 'N';
    case CHESS_WB:
    case CHESS_BB:
        return 'B';
    case CHESS_WR:
    case CHESS_BR:
        return 'R';
    case CHESS_WQ:
    case CHESS_BQ:
        return 'Q';
    case CHESS_WK:
    case CHESS_BK:
        return 'K';
    default:
        return 'P';
    }
}

char Piece::get_piece_fr_pgn_piece(char pgn_piece, bool white)
{
    switch (pgn_piece)
    {
    case 'N':
        if (white)
        {
            return CHESS_WN;
        }
        else
        {
            return CHESS_BN;
        }
    case 'B':
        if (white)
        {
            return CHESS_WB;
        }
        else
        {
            return CHESS_BB;
        }
    case 'R':
        if (white)
        {
            return CHESS_WR;
        }
        else
        {
            return CHESS_BR;
        }
    case 'Q':
        if (white)
        {
            return CHESS_WQ;
        }
        else
        {
            return CHESS_BQ;
        }
    case 'K':
        if (white)
        {
            return CHESS_WK;
        }
        else
        {
            return CHESS_BK;
        }
    default:
        // Pawn will be 'P' (optional).
        if (white)
        {
            return CHESS_WP;
        }
        else
        {
            return CHESS_BP;
        }
    }
}

Board::Board()
{
    this->reset();
}

Board::~Board() {}

int Board::get_row(int square)
{
    return square / CHESS_COL_CNT;
}

int Board::get_row(char alpha_row)
{
    return (alpha_row - '0') - 1;
}

int Board::get_col(int square)
{
    return square % CHESS_COL_CNT;
}

int Board::get_col(char alpha_col)
{
    switch (alpha_col)
    {
    case 'a':
        return 0;
    case 'b':
        return 1;
    case 'c':
        return 2;
    case 'd':
        return 3;
    case 'e':
        return 4;
    case 'f':
        return 5;
    case 'g':
        return 6;
    default:
        return 7;
    }
}

char Board::get_alpha_col(int col)
{
    switch (col)
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

int Board::get_square(int row, int col)
{
    return row * CHESS_COL_CNT + col;
}

int Board::get_square(int row, char alpha_col)
{
    return row * CHESS_COL_CNT + Board::get_col(alpha_col);
}

int Board::get_square(char alpha_row, char alpha_col)
{
    return Board::get_row(alpha_row) * CHESS_COL_CNT + Board::get_col(alpha_col);
}

bool Board::is_row_valid(int row)
{
    return row >= 0 && row < CHESS_ROW_CNT;
}

bool Board::is_col_valid(int col)
{
    return col >= 0 && col < CHESS_COL_CNT;
}

void Board::reset()
{
    memcpy(this->data_, CHESS_BOARD_START_STATE, sizeof(char) * CHESS_BOARD_LEN);

    memset(&this->castle_state_, 0, sizeof(this->castle_state_));
    memset(&this->check_state_, 0, sizeof(this->check_state_));
}

void Board::copy(Board *src)
{
    memcpy(this->data_, src->data_, sizeof(char) * CHESS_BOARD_LEN);
    this->castle_state_ = src->castle_state_;
    this->check_state_ = src->check_state_;
}

char *Board::get_data()
{
    return this->data_;
}

int Board::compare_data(Board *other)
{
    return memcmp(this->data_, other->data_, sizeof(this->data_));
}

int Board::compare_data(const char *other_data)
{
    return memcmp(this->data_, other_data, sizeof(this->data_));
}

void Board::print()
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    printf("\n");

    bool white_first = true;

    int foreground = 0;
    int background = 0;

    for (int row = CHESS_ROW_CNT - 1; row >= 0; row--)
    {
        printf("%d  ", row + 1);
        printf("");

        for (int col = 0; col < CHESS_COL_CNT; col++)
        {
            char piece = this->get_piece(Board::get_square(row, col));

            if (Piece::is_white(piece))
            {
                foreground = 15;
            }
            else if (Piece::is_black(piece))
            {
                foreground = 0;
            }
            else
            {
                foreground = 15;
            }

            if (col % 2 == 0)
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

            printf("%s", Piece::to_str(piece));
        }

        white_first = !white_first;

        FlushConsoleInputBuffer(hConsole);
        SetConsoleTextAttribute(hConsole, 15);

        printf("\n");
    }

    FlushConsoleInputBuffer(hConsole);
    SetConsoleTextAttribute(hConsole, 15);

    printf("   ");
    for (int col = 0; col < CHESS_COL_CNT; col++)
    {
        printf(" %c ", Board::get_alpha_col(col));
    }

    printf("\n\n");
}

void Board::print(Move move)
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    printf("\n");

    bool white_first = true;

    int foreground = 0;
    int background = 0;

    for (int row = CHESS_ROW_CNT - 1; row >= 0; row--)
    {
        printf("%d  ", row + 1);
        printf("");

        for (int col = 0; col < CHESS_COL_CNT; col++)
        {
            char piece = this->get_piece(Board::get_square(row, col));

            int square = this->get_square(row, col);

            if (Piece::is_white(piece))
            {
                foreground = 15;
            }
            else if (Piece::is_black(piece))
            {
                foreground = 0;
            }
            else
            {
                foreground = 15;
            }

            if (col % 2 == 0)
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

            if (square == move.src_square || square == move.dst_square)
            {
                background = 4;
            }

            FlushConsoleInputBuffer(hConsole);
            SetConsoleTextAttribute(hConsole, foreground + background * 16);

            printf("%s", Piece::to_str(piece));
        }

        white_first = !white_first;

        FlushConsoleInputBuffer(hConsole);
        SetConsoleTextAttribute(hConsole, 15);

        printf("\n");
    }

    FlushConsoleInputBuffer(hConsole);
    SetConsoleTextAttribute(hConsole, 15);

    printf("   ");
    for (int col = 0; col < CHESS_COL_CNT; col++)
    {
        printf(" %c ", Board::get_alpha_col(col));
    }

    printf("\n\n");
}

char Board::get_piece(int square)
{
    return this->data_[square];
}

void Board::update_diagonal_pins(int square)
{
    int row = Board::get_row(square);
    int col = Board::get_col(square);
    char piece = this->get_piece(square);
    bool white = Piece::is_white(piece);

    int cnt;
    switch (piece)
    {
    case CHESS_WB:
    case CHESS_BB:
    case CHESS_WQ:
    case CHESS_BQ:
        cnt = 8;
        break;
    case CHESS_WK:
    case CHESS_BK:
        cnt = 2;
        break;
    default:
        break;
    }

    int test_square;
    int test_row;
    int test_col;
    int test_pin_square;

    bool ne = false;
    bool sw = false;
    bool se = false;
    bool nw = false;

    int opp_king_square = this->get_king_square(!white);
    bool opp_king_white = Piece::is_white(this->get_piece(opp_king_square));

    // Northeast.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row + i;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!ne)
            {
                if (test_piece != CHESS_MT)
                {
                    test_pin_square = test_square;
                    ne = true;
                }
            }
            else
            {
                if (test_square == opp_king_square)
                {
                    if (opp_king_white)
                    {
                        this->check_state_.white_king_pins[test_pin_square] = true;
                    }
                    else
                    {
                        this->check_state_.black_king_pins[test_pin_square] = true;
                    }
                    break;
                }
                else if (test_piece != CHESS_MT)
                {
                    break;
                }
            }
        }
    }

    // Southwest.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row - i;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!sw)
            {
                if (test_piece != CHESS_MT)
                {
                    test_pin_square = test_square;
                    sw = true;
                }
            }
            else
            {
                if (test_square == opp_king_square)
                {
                    if (opp_king_white)
                    {
                        this->check_state_.white_king_pins[test_pin_square] = true;
                    }
                    else
                    {
                        this->check_state_.black_king_pins[test_pin_square] = true;
                    }
                    break;
                }
                else if (test_piece != CHESS_MT)
                {
                    break;
                }
            }
        }
    }

    // Southeast.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row - i;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!se)
            {
                if (test_piece != CHESS_MT)
                {
                    test_pin_square = test_square;
                    se = true;
                }
            }
            else
            {
                if (test_square == opp_king_square)
                {
                    if (opp_king_white)
                    {
                        this->check_state_.white_king_pins[test_pin_square] = true;
                    }
                    else
                    {
                        this->check_state_.black_king_pins[test_pin_square] = true;
                    }
                    break;
                }
                else if (test_piece != CHESS_MT)
                {
                    break;
                }
            }
        }
    }

    // Northwest.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row + i;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!nw)
            {
                if (test_piece != CHESS_MT)
                {
                    test_pin_square = test_square;
                    nw = true;
                }
            }
            else
            {
                if (test_square == opp_king_square)
                {
                    if (opp_king_white)
                    {
                        this->check_state_.white_king_pins[test_pin_square] = true;
                    }
                    else
                    {
                        this->check_state_.black_king_pins[test_pin_square] = true;
                    }
                    break;
                }
                else if (test_piece != CHESS_MT)
                {
                    break;
                }
            }
        }
    }
}

void Board::update_straight_pins(int square)
{
    int row = Board::get_row(square);
    int col = Board::get_col(square);
    char piece = this->get_piece(square);
    bool white = Piece::is_white(piece);

    int cnt;
    switch (piece)
    {
    case CHESS_WR:
    case CHESS_BR:
    case CHESS_WQ:
    case CHESS_BQ:
        cnt = 8;
        break;
    case CHESS_WK:
    case CHESS_BK:
        cnt = 2;
        break;
    default:
        break;
    }

    int test_square;
    int test_row;
    int test_col;
    int test_pin_square;

    bool n = false;
    bool s = false;
    bool e = false;
    bool w = false;

    int opp_king_square = this->get_king_square(!white);
    bool opp_king_white = Piece::is_white(this->get_piece(opp_king_square));

    // North.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row + i;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!n)
            {
                if (test_piece != CHESS_MT)
                {
                    test_pin_square = test_square;
                    n = true;
                }
            }
            else
            {
                if (test_square == opp_king_square)
                {
                    if (opp_king_white)
                    {
                        this->check_state_.white_king_pins[test_pin_square] = true;
                    }
                    else
                    {
                        this->check_state_.black_king_pins[test_pin_square] = true;
                    }
                    break;
                }
                else if (test_piece != CHESS_MT)
                {
                    break;
                }
            }
        }
    }

    // South.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row - i;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!s)
            {
                if (test_piece != CHESS_MT)
                {
                    test_pin_square = test_square;
                    s = true;
                }
            }
            else
            {
                if (test_square == opp_king_square)
                {
                    if (opp_king_white)
                    {
                        this->check_state_.white_king_pins[test_pin_square] = true;
                    }
                    else
                    {
                        this->check_state_.black_king_pins[test_pin_square] = true;
                    }
                    break;
                }
                else if (test_piece != CHESS_MT)
                {
                    break;
                }
            }
        }
    }

    // East.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!e)
            {
                if (test_piece != CHESS_MT)
                {
                    test_pin_square = test_square;
                    e = true;
                }
            }
            else
            {
                if (test_square == opp_king_square)
                {
                    if (opp_king_white)
                    {
                        this->check_state_.white_king_pins[test_pin_square] = true;
                    }
                    else
                    {
                        this->check_state_.black_king_pins[test_pin_square] = true;
                    }
                    break;
                }
                else if (test_piece != CHESS_MT)
                {
                    break;
                }
            }
        }
    }

    // West.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!w)
            {
                if (test_piece != CHESS_MT)
                {
                    test_pin_square = test_square;
                    w = true;
                }
            }
            else
            {
                if (test_square == opp_king_square)
                {
                    if (opp_king_white)
                    {
                        this->check_state_.white_king_pins[test_pin_square] = true;
                    }
                    else
                    {
                        this->check_state_.black_king_pins[test_pin_square] = true;
                    }
                    break;
                }
                else if (test_piece != CHESS_MT)
                {
                    break;
                }
            }
        }
    }
}

void Board::update_pins(bool white)
{
    if (white)
    {
        memset(this->check_state_.black_king_pins, 0, sizeof(this->check_state_.black_king_pins));

        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            switch (this->get_piece(i))
            {
            case CHESS_WB:
                this->update_diagonal_pins(i);
                break;
            case CHESS_WR:
                this->update_straight_pins(i);
                break;
            case CHESS_WQ:
                this->update_diagonal_pins(i);
                this->update_straight_pins(i);
                break;
            default:
                break;
            }
        }
    }
    else
    {
        memset(this->check_state_.white_king_pins, 0, sizeof(this->check_state_.white_king_pins));

        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            switch (this->get_piece(i))
            {
            case CHESS_BB:
                this->update_diagonal_pins(i);
                break;
            case CHESS_BR:
                this->update_straight_pins(i);
                break;
            case CHESS_BQ:
                this->update_diagonal_pins(i);
                this->update_straight_pins(i);
                break;
            default:
                break;
            }
        }
    }
}

std::vector<Move> Board::get_diagonal_moves(int square, char piece, int row, int col)
{
    std::vector<Move> moves;

    bool white = Piece::is_white(this->get_piece(square));

    int cnt;
    switch (piece)
    {
    case CHESS_WB:
    case CHESS_BB:
    case CHESS_WQ:
    case CHESS_BQ:
        cnt = 8;
        break;
    case CHESS_WK:
    case CHESS_BK:
        cnt = 2;
        break;
    default:
        return moves;
    }

    int test_square;
    int test_row;
    int test_col;

    bool ne = false;
    bool sw = false;
    bool se = false;
    bool nw = false;

    // Northeast.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row + i;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!ne)
            {
                if (test_piece == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

                    ne = true;
                }
            }
        }
    }

    // Southwest.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row - i;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!sw)
            {
                if (test_piece == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

                    sw = true;
                }
            }
        }
    }

    // Southeast.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row - i;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!se)
            {
                if (test_piece == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

                    se = true;
                }
            }
        }
    }

    // Northwest.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row + i;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!nw)
            {
                if (test_piece == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

                    nw = true;
                }
            }
        }
    }

    return moves;
}

std::vector<Move> Board::get_straight_moves(int square, char piece, int row, int col)
{
    std::vector<Move> moves;

    bool white = Piece::is_white(this->get_piece(square));

    int cnt;
    switch (piece)
    {
    case CHESS_WR:
    case CHESS_BR:
    case CHESS_WQ:
    case CHESS_BQ:
        cnt = 8;
        break;
    case CHESS_WK:
    case CHESS_BK:
        cnt = 2;
        break;
    default:
        return moves;
    }

    int test_square;
    int test_row;
    int test_col;

    bool n = false;
    bool s = false;
    bool e = false;
    bool w = false;

    // North.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row + i;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!n)
            {
                if (test_piece == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

                    n = true;
                }
            }
        }
    }

    // South.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row - i;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!s)
            {
                if (test_piece == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

                    s = true;
                }
            }
        }
    }

    // East.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!e)
            {
                if (test_piece == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

                    e = true;
                }
            }
        }
    }

    // West.
    for (int i = 1; i < cnt; i++)
    {
        test_row = row;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

            if (!w)
            {
                if (test_piece == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

                    w = true;
                }
            }
        }
    }

    return moves;
}

std::vector<Move> Board::get_moves(int square, bool test_check)
{
    std::vector<Move> moves;

    char piece = this->get_piece(square);
    bool white = Piece::is_white(piece);

    if (piece == CHESS_MT)
    {
        return moves;
    }

    int row = Board::get_row(square);
    int col = Board::get_col(square);

    int test_square;
    int test_row;
    int test_col;

    switch (piece)
    {
    case CHESS_WP:
    {
        test_row = row + 1;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && this->get_piece(test_square) == CHESS_MT)
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row + 1;
        test_col = col - 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) && Piece::is_black(this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row + 1;
        test_col = col + 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) && Piece::is_black(this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        if (row == 1)
        {
            test_row = row + 1;
            test_col = col;
            test_square = Board::get_square(test_row, test_col);

            if (this->get_piece(test_square) == CHESS_MT)
            {
                test_row = row + 2;
                test_col = col;
                test_square = Board::get_square(test_row, test_col);

                if (this->get_piece(test_square) == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
            }
        }

        if (row == 4)
        {
            if (Board::is_col_valid(this->au_passant_state_.dst_col))
            {
                if (this->au_passant_state_.dst_col == col - 1)
                {
                    moves.push_back(Move{square, Board::get_square(row + 1, col - 1)});
                }
                else if (this->au_passant_state_.dst_col == col + 1)
                {
                    moves.push_back(Move{square, Board::get_square(row + 1, col + 1)});
                }
            }
        }
    }
    break;
    case CHESS_BP:
    {
        test_row = row - 1;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && this->get_piece(test_square) == CHESS_MT)
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row - 1;
        test_col = col - 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) && Piece::is_white(this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row - 1;
        test_col = col + 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) && Piece::is_white(this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        if (row == 6)
        {
            test_row = row - 1;
            test_col = col;
            test_square = Board::get_square(test_row, test_col);

            if (this->get_piece(test_square) == CHESS_MT)
            {
                test_row = row - 2;
                test_col = col;
                test_square = Board::get_square(test_row, test_col);

                if (this->get_piece(test_square) == CHESS_MT)
                {
                    moves.push_back(Move{square, test_square});
                }
            }
        }

        if (row == 3)
        {
            if (Board::is_col_valid(this->au_passant_state_.dst_col))
            {
                if (this->au_passant_state_.dst_col == col - 1)
                {
                    moves.push_back(Move{square, Board::get_square(row - 1, col - 1)});
                }
                else if (this->au_passant_state_.dst_col == col + 1)
                {
                    moves.push_back(Move{square, Board::get_square(row - 1, col + 1)});
                }
            }
        }
    }
    break;
    case CHESS_WN:
    case CHESS_BN:
    {
        test_row = row + 2;
        test_col = col + 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) &&
            !Piece::is_same_color(this->get_piece(square), this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row - 2;
        test_col = col + 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) &&
            !Piece::is_same_color(this->get_piece(square), this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row + 2;
        test_col = col - 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) &&
            !Piece::is_same_color(this->get_piece(square), this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row - 2;
        test_col = col - 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) &&
            !Piece::is_same_color(this->get_piece(square), this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row + 1;
        test_col = col + 2;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) &&
            !Piece::is_same_color(this->get_piece(square), this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row - 1;
        test_col = col + 2;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) &&
            !Piece::is_same_color(this->get_piece(square), this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row + 1;
        test_col = col - 2;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) &&
            !Piece::is_same_color(this->get_piece(square), this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row - 1;
        test_col = col - 2;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) &&
            !Piece::is_same_color(this->get_piece(square), this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }
    }
    break;
    case CHESS_WB:
    case CHESS_BB:
    {
        auto diagonal_moves = this->get_diagonal_moves(square, piece, row, col);
        moves.insert(moves.end(), diagonal_moves.begin(), diagonal_moves.end());
    }
    break;
    case CHESS_WR:
    case CHESS_BR:
    {
        auto straight_moves = this->get_straight_moves(square, piece, row, col);
        moves.insert(moves.end(), straight_moves.begin(), straight_moves.end());
    }
    break;
    case CHESS_WQ:
    case CHESS_BQ:
    {
        auto diagonal_moves = this->get_diagonal_moves(square, piece, row, col);
        moves.insert(moves.end(), diagonal_moves.begin(), diagonal_moves.end());

        auto straight_moves = this->get_straight_moves(square, piece, row, col);
        moves.insert(moves.end(), straight_moves.begin(), straight_moves.end());
    }
    break;
    case CHESS_WK:
    case CHESS_BK:
    {
        std::vector<Move> king_moves;

        auto diagonal_moves = this->get_diagonal_moves(square, piece, row, col);
        king_moves.insert(king_moves.end(), diagonal_moves.begin(), diagonal_moves.end());

        auto straight_moves = this->get_straight_moves(square, piece, row, col);
        king_moves.insert(king_moves.end(), straight_moves.begin(), straight_moves.end());

        if (test_check)
        {
            if (piece == CHESS_WK && !this->castle_state_.white_king_moved)
            {
                if (this->get_piece(0) == CHESS_WR && !this->castle_state_.white_left_rook_moved)
                {
                    if (this->get_piece(1) == CHESS_MT && this->get_piece(2) == CHESS_MT && this->get_piece(3) == CHESS_MT)
                    {
                        if (!this->is_square_under_attack(2, false) && !this->is_square_under_attack(3, false))
                        {
                            king_moves.push_back(Move{square, 2});
                        }
                    }
                }

                if (this->get_piece(7) == CHESS_WR && !this->castle_state_.white_right_rook_moved)
                {
                    if (this->get_piece(5) == CHESS_MT && this->get_piece(6) == CHESS_MT)
                    {
                        if (!this->is_square_under_attack(5, false) && !this->is_square_under_attack(6, false))
                        {
                            king_moves.push_back(Move{square, 6});
                        }
                    }
                }
            }
            else if (piece == CHESS_BK && !this->castle_state_.black_king_moved)
            {
                if (this->get_piece(56) == CHESS_BR && !this->castle_state_.black_left_rook_moved)
                {
                    if (this->get_piece(57) == CHESS_MT && this->get_piece(58) == CHESS_MT && this->get_piece(59) == CHESS_MT)
                    {
                        if (!this->is_square_under_attack(58, true) && !this->is_square_under_attack(59, true))
                        {
                            king_moves.push_back(Move{square, 58});
                        }
                    }
                }

                if (this->get_piece(63) == CHESS_BR && !this->castle_state_.black_right_rook_moved)
                {
                    if (this->get_piece(61) == CHESS_MT && this->get_piece(62) == CHESS_MT)
                    {
                        if (!this->is_square_under_attack(61, true) && !this->is_square_under_attack(62, true))
                        {
                            king_moves.push_back(Move{square, 62});
                        }
                    }
                }
            }

            // Make sure king is not moving into check.
            {
                std::vector<Move> tested_moves;

                for (auto move : king_moves)
                {
                    auto sim = this->simulate(move);

                    if (!sim.board.is_check(!white, true))
                    {
                        tested_moves.push_back(move);
                    }
                }

                king_moves = tested_moves;
            }
        }

        moves.insert(moves.end(), king_moves.begin(), king_moves.end());
    }
    break;
    default:
        break;
    }

    if (test_check)
    {
        if (white)
        {
            if (this->check_state_.white_checked)
            {
                std::vector<Move> tested_moves;

                for (auto move : moves)
                {
                    auto sim = this->simulate(move);

                    if (!sim.board.is_check(!white, true))
                    {
                        tested_moves.push_back(move);
                    }
                }

                moves = tested_moves;
            }
            else
            {
                if (this->is_piece_in_king_pin(square, white))
                {
                    std::vector<Move> tested_moves;

                    for (auto move : moves)
                    {
                        auto sim = this->simulate(move);

                        if (!sim.board.is_check(!white, true))
                        {
                            tested_moves.push_back(move);
                        }
                    }

                    moves = tested_moves;
                }
            }
        }
        else
        {
            if (this->check_state_.black_checked)
            {
                std::vector<Move> tested_moves;

                for (auto move : moves)
                {
                    auto sim = this->simulate(move);

                    if (!sim.board.is_check(!white, true))
                    {
                        tested_moves.push_back(move);
                    }
                }

                moves = tested_moves;
            }
            else
            {
                if (this->is_piece_in_king_pin(square, white))
                {
                    std::vector<Move> tested_moves;

                    for (auto move : moves)
                    {
                        auto sim = this->simulate(move);

                        if (!sim.board.is_check(!white, true))
                        {
                            tested_moves.push_back(move);
                        }
                    }

                    moves = tested_moves;
                }
            }
        }
    }

    return moves;
}

std::vector<Move> Board::get_all_moves(bool white)
{
    std::vector<Move> all_moves;

    if (white)
    {
        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            if (Piece::is_white(this->get_piece(i)))
            {
                auto moves = this->get_moves(i, true);
                all_moves.insert(all_moves.end(), moves.begin(), moves.end());
            }
        }
    }
    else
    {
        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            if (Piece::is_black(this->get_piece(i)))
            {
                auto moves = this->get_moves(i, true);
                all_moves.insert(all_moves.end(), moves.begin(), moves.end());
            }
        }
    }

    return all_moves;
}

bool Board::has_moves(bool white)
{
    if (white)
    {
        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            if (Piece::is_white(this->get_piece(i)))
            {
                auto moves = this->get_moves(i, true);
                if (moves.size() > 0)
                {
                    return true;
                }
            }
        }
    }
    else
    {
        for (int i = CHESS_BOARD_LEN - 1; i >= 0; i--)
        {
            if (Piece::is_black(this->get_piece(i)))
            {
                auto moves = this->get_moves(i, true);
                if (moves.size() > 0)
                {
                    return true;
                }
            }
        }
    }

    return false;
}

bool Board::is_square_under_attack(int square, bool by_white)
{
    if (by_white)
    {
        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            if (Piece::is_white(this->get_piece(i)))
            {
                auto moves = this->get_moves(i, false);
                for (auto move : moves)
                {
                    if (move.dst_square == square)
                    {
                        return true;
                    }
                }
            }
        }
    }
    else
    {
        for (int i = CHESS_BOARD_LEN - 1; i >= 0; i--)
        {
            if (Piece::is_black(this->get_piece(i)))
            {
                auto moves = this->get_moves(i, false);
                for (auto move : moves)
                {
                    if (move.dst_square == square)
                    {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

int Board::get_king_square(bool white)
{
    int king_square = -1;
    if (white)
    {
        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            if (this->get_piece(i) == CHESS_WK)
            {
                king_square = i;
                break;
            }
        }
    }
    else
    {
        for (int i = CHESS_BOARD_LEN - 1; i >= 0; i--)
        {
            if (this->get_piece(i) == CHESS_BK)
            {
                king_square = i;
                break;
            }
        }
    }

    return king_square;
}

bool Board::is_piece_in_king_pin(int square, bool white_king_pin)
{
    if (white_king_pin)
    {
        return this->check_state_.white_king_pins[square];
    }
    else
    {
        return this->check_state_.black_king_pins[square];
    }

    return false;
}

bool Board::is_check(bool by_white, bool hard_way)
{
    if (hard_way)
    {
        if (by_white)
        {
            return this->is_square_under_attack(this->get_king_square(false), by_white);
        }
        else
        {
            return this->is_square_under_attack(this->get_king_square(true), by_white);
        }
    }
    else
    {
        if (by_white)
        {
            return this->check_state_.black_checked;
        }
        else
        {
            return this->check_state_.white_checked;
        }
    }

    return false;
}

bool Board::is_check(bool by_white)
{
    return this->is_check(by_white, true);
}

bool Board::is_checkmate(bool by_white, bool hard_way)
{
    if (this->is_check(by_white, hard_way))
    {
        if (!this->has_moves(!by_white))
        {
            return true;
        }
    }

    return false;
}

bool Board::is_checkmate(bool by_white)
{
    return this->is_checkmate(by_white, true);
}

void Board::change(Move move)
{
    if (!Move::is_valid(&move))
    {
        CHESS_THROW_ERROR("CHESS ERROR: invalid move");
    }

    char src_piece = this->get_piece(move.src_square);
    char dst_piece = src_piece;

    int src_row = Board::get_row(move.src_square);
    int src_col = Board::get_col(move.src_square);
    int dst_row = Board::get_row(move.dst_square);
    int dst_col = Board::get_col(move.dst_square);

    bool white = Piece::is_white(src_piece);

    this->au_passant_state_.dst_col = CHESS_INVALID_SQUARE;

    switch (src_piece)
    {
    case CHESS_WP:
    {
        // Look for promotion and au passant.

        if (dst_row == 7)
        {
            if (move.promo_piece == CHESS_MT)
            {
                dst_piece = CHESS_WQ;
            }
            else
            {
                dst_piece = move.promo_piece;
            }
        }
        else if (dst_row == 5)
        {
            if (this->get_piece(move.dst_square) == CHESS_MT)
            {
                int test_au_passant_square = Board::get_square(dst_row - 1, dst_col);
                if (this->get_piece(test_au_passant_square) == CHESS_BP)
                {
                    this->data_[test_au_passant_square] = CHESS_MT;
                }
            }
        }
        else
        {
            if (abs(dst_row - src_row) == 2)
            {
                this->au_passant_state_.dst_col = dst_col;
            }
        }
    }
    break;
    case CHESS_BP:
    {
        // Look for promotion and au passant.

        if (dst_row == 0)
        {
            if (move.promo_piece == CHESS_MT)
            {
                dst_piece = CHESS_BQ;
            }
            else
            {
                dst_piece = move.promo_piece;
            }
        }
        else if (dst_row == 2)
        {
            if (this->get_piece(move.dst_square) == CHESS_MT)
            {
                int test_au_passant_square = Board::get_square(dst_row + 1, dst_col);
                if (this->get_piece(test_au_passant_square) == CHESS_WP)
                {
                    this->data_[test_au_passant_square] = CHESS_MT;
                }
            }
        }
        else
        {
            if (abs(dst_row - src_row) == 2)
            {
                this->au_passant_state_.dst_col = dst_col;
            }
        }
    }
    break;
    case CHESS_WR:
    {
        if (src_row == 0 && src_col == 0)
        {
            this->castle_state_.white_left_rook_moved = true;
        }
        else if (src_row == 0 && src_col == 7)
        {
            this->castle_state_.white_right_rook_moved = true;
        }
    }
    break;
    case CHESS_BR:
    {
        if (src_row == 7 && src_col == 0)
        {
            this->castle_state_.black_left_rook_moved = true;
        }
        else if (src_row == 7 && src_col == 7)
        {
            this->castle_state_.black_right_rook_moved = true;
        }
    }
    break;
    case CHESS_WK:
    {
        // Look for castle.

        if (src_col - dst_col == 2)
        {
            this->data_[0] = CHESS_MT;
            this->data_[3] = CHESS_WR;

            // Check new moves for moved rook.
            {
                auto moves = this->get_moves(3, false);
                int opp_king_square = this->get_king_square(!white);
                for (auto move : moves)
                {
                    if (move.dst_square == opp_king_square)
                    {
                        if (white)
                        {
                            this->check_state_.black_checked = true;
                        }
                        else
                        {
                            this->check_state_.white_checked = true;
                        }
                        break;
                    }
                }
            }
        }
        else if (src_col - dst_col == -2)
        {
            this->data_[7] = CHESS_MT;
            this->data_[5] = CHESS_WR;

            // Check new moves for moved rook.
            {
                auto moves = this->get_moves(5, false);
                int opp_king_square = this->get_king_square(!white);
                for (auto move : moves)
                {
                    if (move.dst_square == opp_king_square)
                    {
                        if (white)
                        {
                            this->check_state_.black_checked = true;
                        }
                        else
                        {
                            this->check_state_.white_checked = true;
                        }
                        break;
                    }
                }
            }
        }

        this->castle_state_.white_king_moved = true;
    }
    break;
    case CHESS_BK:
    {
        // Look for castle.

        if (src_col - dst_col == 2)
        {
            this->data_[56] = CHESS_MT;
            this->data_[59] = CHESS_BR;

            // Check new moves for moved rook.
            {
                auto moves = this->get_moves(59, false);
                int opp_king_square = this->get_king_square(!white);
                for (auto move : moves)
                {
                    if (move.dst_square == opp_king_square)
                    {
                        if (white)
                        {
                            this->check_state_.black_checked = true;
                        }
                        else
                        {
                            this->check_state_.white_checked = true;
                        }
                        break;
                    }
                }
            }
        }
        else if (src_col - dst_col == -2)
        {
            this->data_[63] = CHESS_MT;
            this->data_[61] = CHESS_BR;

            // Check new moves for moved rook.
            {
                auto moves = this->get_moves(61, false);
                int opp_king_square = this->get_king_square(!white);
                for (auto move : moves)
                {
                    if (move.dst_square == opp_king_square)
                    {
                        if (white)
                        {
                            this->check_state_.black_checked = true;
                        }
                        else
                        {
                            this->check_state_.white_checked = true;
                        }
                        break;
                    }
                }
            }
        }

        this->castle_state_.black_king_moved = true;
    }
    break;
    default:
        break;
    }

    this->data_[move.src_square] = CHESS_MT;
    this->data_[move.dst_square] = dst_piece;

    this->check_state_.white_checked = false;
    this->check_state_.black_checked = false;

    // Check new moves for moved piece.
    auto moves = this->get_moves(move.dst_square, false);
    int opp_king_square = this->get_king_square(!white);
    for (auto move : moves)
    {
        if (move.dst_square == opp_king_square)
        {
            if (white)
            {
                this->check_state_.black_checked = true;
            }
            else
            {
                this->check_state_.white_checked = true;
            }
            break;
        }
    }

    // Check if piece was pinned to opponent king.
    if (white)
    {
        if (this->is_piece_in_king_pin(move.src_square, false))
        {
            this->check_state_.black_checked = true;
        }
    }
    else
    {
        if (this->is_piece_in_king_pin(move.src_square, true))
        {
            this->check_state_.white_checked = true;
        }
    }

    // New pin is possible as a result of moving piece.
    this->update_pins(white);
}

Move Board::change(std::string move_str, bool white)
{
    this->update_pins(white);

    char piece;
    char promo_piece = CHESS_MT;

    int src_row;
    int src_col;
    int dst_row;
    int dst_col;

    int src_square = CHESS_INVALID_SQUARE;
    int dst_square = CHESS_INVALID_SQUARE;

    std::string mut_move_str = move_str;

    // Trim '+'/'#'.
    mut_move_str.erase(remove(mut_move_str.begin(), mut_move_str.end(), '+'), mut_move_str.end());
    mut_move_str.erase(remove(mut_move_str.begin(), mut_move_str.end(), '#'), mut_move_str.end());

    // Remove 'x'.
    mut_move_str.erase(remove(mut_move_str.begin(), mut_move_str.end(), 'x'), mut_move_str.end());

    switch (mut_move_str.size())
    {
    case 2:
    {
        // Pawn move.

        dst_row = Board::get_row(mut_move_str[1]);
        dst_col = Board::get_col(mut_move_str[0]);
        dst_square = Board::get_square(dst_row, dst_col);

        if (white)
        {
            int prev_square = Board::get_square(dst_row - 1, dst_col);
            int prev_square_2 = Board::get_square(dst_row - 2, dst_col);

            if (this->get_piece(prev_square) == CHESS_WP)
            {
                src_square = prev_square;
            }
            else if (this->get_piece(prev_square_2) == CHESS_WP)
            {
                src_square = prev_square_2;
            }
        }
        else
        {
            int prev_square = Board::get_square(dst_row + 1, dst_col);
            int prev_square_2 = Board::get_square(dst_row + 2, dst_col);

            if (this->get_piece(prev_square) == CHESS_BP)
            {
                src_square = prev_square;
            }
            else if (this->get_piece(prev_square_2) == CHESS_BP)
            {
                src_square = prev_square_2;
            }
        }
    }
    break;
    case 3:
    {
        if (mut_move_str.compare("O-O") == 0)
        {
            if (white)
            {
                src_square = 4;
                dst_square = 6;
            }
            else
            {
                src_square = 60;
                dst_square = 62;
            }
        }
        else
        {
            // Need to check if isupper since pawn move will not have piece id -- just the src col.
            if (isupper(mut_move_str[0]) == 1)
            {
                // Minor/major piece move.
                piece = Piece::get_piece_fr_pgn_piece(mut_move_str[0], white);
                dst_row = Board::get_row(mut_move_str[2]);
                dst_col = Board::get_col(mut_move_str[1]);
                dst_square = Board::get_square(dst_row, dst_col);

                // It is possible that move will not be properly disambiguated. Using piece closest to destination may help.
                int dist = INT_MAX;

                for (int square = 0; square < CHESS_BOARD_LEN; square++)
                {
                    if (this->get_piece(square) == piece)
                    {
                        auto moves = this->get_moves(square, true);
                        for (auto move : moves)
                        {
                            if (move.dst_square == dst_square)
                            {
                                src_row = Board::get_row(square);
                                src_col = Board::get_col(square);

                                int cur_dist = ((dst_row - src_row) * (dst_row - src_row)) + ((dst_col - src_col) * (dst_col - src_col));
                                if (cur_dist < dist)
                                {
                                    dist = cur_dist;
                                    src_square = square;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // Disambiguated pawn move.
                src_col = Board::get_col(mut_move_str[0]);
                dst_row = Board::get_row(mut_move_str[2]);
                dst_col = Board::get_col(mut_move_str[1]);
                dst_square = Board::get_square(dst_row, dst_col);

                if (white)
                {
                    src_square = Board::get_square(dst_row - 1, src_col);
                }
                else
                {
                    src_square = Board::get_square(dst_row + 1, src_col);
                }
            }
        }
    }
    break;
    case 4:
    {
        // Need to check if isupper since pawn move will not have piece id -- just the src col.
        if (isupper(mut_move_str[0]) == 1)
        {
            // Disambiguated minor/major piece move.
            piece = Piece::get_piece_fr_pgn_piece(mut_move_str[0], white);
            dst_row = Board::get_row(mut_move_str[3]);
            dst_col = Board::get_col(mut_move_str[2]);

            if (isdigit(mut_move_str[1]))
            {
                src_row = Board::get_row(mut_move_str[1]);
                dst_square = Board::get_square(dst_row, dst_col);

                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (Board::get_row(i) == src_row && this->get_piece(i) == piece)
                    {
                        bool legal_move = false;
                        auto moves = this->get_moves(i, true);
                        for (auto move : moves)
                        {
                            if (move.dst_square == dst_square)
                            {
                                legal_move = true;
                                break;
                            }
                        }

                        if (legal_move)
                        {
                            src_square = i;
                            break;
                        }
                    }
                }
            }
            else
            {
                src_col = Board::get_col(mut_move_str[1]);
                dst_square = Board::get_square(dst_row, dst_col);

                for (int i = 0; i < CHESS_BOARD_LEN; i++)
                {
                    if (Board::get_col(i) == src_col && this->get_piece(i) == piece)
                    {
                        bool legal_move = false;
                        auto moves = this->get_moves(i, true);
                        for (auto move : moves)
                        {
                            if (move.dst_square == dst_square)
                            {
                                legal_move = true;
                                break;
                            }
                        }

                        if (legal_move)
                        {
                            src_square = i;
                            break;
                        }
                    }
                }
            }
        }
        else
        {
            // Pawn promotion.
            if (mut_move_str[2] == '=')
            {
                dst_row = Board::get_row(mut_move_str[1]);
                dst_col = Board::get_col(mut_move_str[0]);
                dst_square = Board::get_square(dst_row, dst_col);
                piece = Piece::get_piece_fr_pgn_piece(mut_move_str[3], white);
                promo_piece = piece;

                if (white)
                {
                    piece = CHESS_WP;
                }
                else
                {
                    piece = CHESS_BP;
                }

                bool found = false;
                for (int square = 0; square < CHESS_BOARD_LEN; square++)
                {
                    if (this->get_piece(square) == piece)
                    {
                        auto moves = this->get_moves(square, true);
                        for (auto move : moves)
                        {
                            if (move.dst_square == dst_square)
                            {
                                src_square = square;
                                found = true;
                                break;
                            }
                        }

                        if (found)
                        {
                            break;
                        }
                    }
                }
            }
        }
    }
    break;
    case 5:
    {
        if (mut_move_str.compare("O-O-O") == 0)
        {
            if (white)
            {
                src_square = 4;
                dst_square = 2;
            }
            else
            {
                src_square = 60;
                dst_square = 58;
            }
        }
        else
        {
            // Need to check if isupper since pawn move will not have piece id -- just the src col.
            if (isupper(mut_move_str[0]) == 1)
            {
                // Disambiguated queen move.
                piece = Piece::get_piece_fr_pgn_piece(mut_move_str[0], white);
                if (piece == CHESS_WQ || piece == CHESS_BQ)
                {
                    src_row = Board::get_row(mut_move_str[2]);
                    src_col = Board::get_col(mut_move_str[1]);
                    dst_row = Board::get_row(mut_move_str[4]);
                    dst_col = Board::get_col(mut_move_str[3]);

                    src_square = Board::get_square(src_row, src_col);
                    dst_square = Board::get_square(dst_row, dst_col);
                }
            }
            else
            {
                // Disambiguated pawn promotion.
                if (mut_move_str[3] == '=')
                {
                    src_col = Board::get_col(mut_move_str[0]);
                    dst_row = Board::get_row(mut_move_str[2]);
                    dst_col = Board::get_col(mut_move_str[1]);
                    dst_square = Board::get_square(dst_row, dst_col);
                    promo_piece = Piece::get_piece_fr_pgn_piece(mut_move_str[4], white);

                    if (white)
                    {
                        src_row = dst_row - 1;
                    }
                    else
                    {
                        src_row = dst_row + 1;
                    }

                    src_square = Board::get_square(src_row, src_col);
                }
            }
        }
    }
    break;
    case 7:
    {
        src_row = Board::get_row(move_str[2]);
        src_col = Board::get_col(move_str[1]);
        src_square = Board::get_square(src_row, src_col);
        dst_row = Board::get_row(move_str[4]);
        dst_col = Board::get_col(move_str[3]);
        dst_square = Board::get_square(dst_row, dst_col);
    }
    break;
    default:
        break;
    }

    Move move{src_square, dst_square, promo_piece};

    // Make sure our engine thinks move is valid.
    {
        bool valid_move = false;

        auto moves = this->get_moves(src_square, true);
        for (auto move : moves)
        {
            if (move.dst_square == dst_square)
            {
                valid_move = true;
                break;
            }
        }

        if (!valid_move)
        {
            this->print();
            printf("INVALID MOVE: %s\n", move_str.c_str());
            return Move{CHESS_INVALID_SQUARE, CHESS_INVALID_SQUARE, CHESS_MT};
        }
    }

    this->change(move);

    return move;
}

Simulation Board::simulate(Move move)
{
    Simulation sim;

    sim.move = move;
    sim.board.copy(this);
    sim.board.change(move);

    return sim;
}

Simulation Board::simulate(std::string move_str, bool white)
{
    Simulation sim;

    sim.board.copy(this);
    sim.move = sim.board.change(move_str, white);

    return sim;
}

std::vector<Simulation> Board::simulate_all(bool white)
{
    std::vector<Simulation> sims;

    auto all_moves = this->get_all_moves(white);

    int move_idx = 0;

    for (auto move : all_moves)
    {
        auto sim = this->simulate(move);
        sim.idx = move_idx;
        sims.push_back(sim);

        move_idx++;
    }

    return sims;
}

int Board::evaluate_material()
{
    int mat_eval = 0;

    for (int i = 0; i < CHESS_BOARD_LEN; i++)
    {
        mat_eval += Piece::get_value(this->get_piece(i));
    }

    return mat_eval;
}

int Board::sim_minimax_alphabeta_sync(Simulation sim, bool white, int depth, int alpha, int beta)
{
    if (sim.board.is_checkmate(!white, false))
    {
        if (white)
        {
            return CHESS_EVAL_MAX_VAL;
        }
        else
        {
            return CHESS_EVAL_MIN_VAL;
        }
    }

    if (depth == 0)
    {
        return sim.board.evaluate_material();
    }

    if (!white)
    {
        int best_eval_val = CHESS_EVAL_MIN_VAL;
        auto sim_sims = sim.board.simulate_all(true);

        for (auto sim_sim : sim_sims)
        {
            int eval_val = Board::sim_minimax_alphabeta_sync(sim_sim, true, depth - 1, alpha, beta);

            best_eval_val = eval_val > best_eval_val ? eval_val : best_eval_val;

            alpha = eval_val > alpha ? eval_val : alpha;
            if (beta <= alpha)
            {
                break;
            }
        }

        return best_eval_val;
    }
    else
    {
        int best_eval_val = CHESS_EVAL_MAX_VAL;
        auto sim_sims = sim.board.simulate_all(false);

        for (auto sim_sim : sim_sims)
        {
            int eval_val = Board::sim_minimax_alphabeta_sync(sim_sim, false, depth - 1, alpha, beta);

            best_eval_val = eval_val < best_eval_val ? eval_val : best_eval_val;

            beta = eval_val < beta ? eval_val : beta;
            if (beta <= alpha)
            {
                break;
            }
        }

        return best_eval_val;
    }
}

void Board::sim_minimax_alphabeta_async(Simulation sim, bool white, int depth, int alpha, int beta, Evaluation *evals)
{
    int eval_val = Board::sim_minimax_alphabeta_sync(sim, white, depth, alpha, beta);
    evals[sim.idx] = Evaluation{eval_val, sim.move, sim.board};
}

std::vector<Evaluation> Board::minimax_alphabeta(bool white, int depth)
{
    std::vector<Evaluation> best_moves;

    Evaluation evals[CHESS_BOARD_LEN];

    auto sims = this->simulate_all(white);

    int min = CHESS_EVAL_MIN_VAL;
    int max = CHESS_EVAL_MAX_VAL;

    int best_eval_val = white ? min : max;

    std::vector<std::thread> threads;

    for (auto sim : sims)
    {
        threads.push_back(std::thread(Board::sim_minimax_alphabeta_async, sim, white, depth, min, max, evals));
    }

    for (auto &th : threads)
    {
        th.join();
    }

    for (int i = 0; i < sims.size(); i++)
    {
        auto eval = evals[i];

        if ((white && eval.value > best_eval_val) || (!white && eval.value < best_eval_val))
        {
            best_moves.clear();

            best_eval_val = eval.value;

            best_moves.push_back(eval);
        }
        else if (eval.value == best_eval_val)
        {
            best_moves.push_back(eval);
        }
    }

    return best_moves;
}

int Board::sim_dyn_minimax_alphabeta_dyn_sync(Simulation sim, bool white, int depth, int alpha, int beta, int depth_inc_cnt)
{
    if (sim.board.is_checkmate(!white, false))
    {
        if (white)
        {
            return CHESS_EVAL_MAX_VAL;
        }
        else
        {
            return CHESS_EVAL_MIN_VAL;
        }
    }

    if (depth == 0)
    {
        return sim.board.evaluate_material();
    }

    int max_depth_inc_cnt = 7;

    if (!white)
    {
        int best_eval_val = CHESS_EVAL_MIN_VAL;
        auto sim_sims = sim.board.simulate_all(true);

        if (sim_sims.size() <= 10 && depth_inc_cnt < max_depth_inc_cnt)
        {
            depth++;
            depth_inc_cnt++;
        }

        for (auto sim_sim : sim_sims)
        {
            int eval_val = Board::sim_dyn_minimax_alphabeta_dyn_sync(sim_sim, true, depth - 1, alpha, beta, depth_inc_cnt);

            best_eval_val = eval_val > best_eval_val ? eval_val : best_eval_val;

            alpha = eval_val > alpha ? eval_val : alpha;
            if (beta <= alpha)
            {
                break;
            }
        }

        return best_eval_val;
    }
    else
    {
        int best_eval_val = CHESS_EVAL_MAX_VAL;
        auto sim_sims = sim.board.simulate_all(false);

        if (sim_sims.size() <= 10 && depth_inc_cnt < max_depth_inc_cnt)
        {
            depth++;
            depth_inc_cnt++;
        }

        for (auto sim_sim : sim_sims)
        {
            int eval_val = Board::sim_dyn_minimax_alphabeta_dyn_sync(sim_sim, false, depth - 1, alpha, beta, depth_inc_cnt);

            best_eval_val = eval_val < best_eval_val ? eval_val : best_eval_val;

            beta = eval_val < beta ? eval_val : beta;
            if (beta <= alpha)
            {
                break;
            }
        }

        return best_eval_val;
    }
}

void Board::sim_minimax_alphabeta_dyn_async(Simulation sim, bool white, int depth, int alpha, int beta, Evaluation *evals)
{
    int eval_val = Board::sim_dyn_minimax_alphabeta_dyn_sync(sim, white, depth, alpha, beta, 0);
    evals[sim.idx] = Evaluation{eval_val, sim.move, sim.board};
}

std::vector<Evaluation> Board::minimax_alphabeta_dyn(bool white, int depth)
{
    std::vector<Evaluation> best_moves;

    Evaluation evals[CHESS_BOARD_LEN];

    auto sims = this->simulate_all(white);

    int min = CHESS_EVAL_MIN_VAL;
    int max = CHESS_EVAL_MAX_VAL;

    int best_eval_val = white ? min : max;

    std::vector<std::thread> threads;

    for (auto sim : sims)
    {
        threads.push_back(std::thread(Board::sim_minimax_alphabeta_dyn_async, sim, white, depth, min, max, evals));
    }

    for (auto &th : threads)
    {
        th.join();
    }

    for (int i = 0; i < sims.size(); i++)
    {
        auto eval = evals[i];

        if ((white && eval.value > best_eval_val) || (!white && eval.value < best_eval_val))
        {
            best_moves.clear();

            best_eval_val = eval.value;

            best_moves.push_back(eval);
        }
        else if (eval.value == best_eval_val)
        {
            best_moves.push_back(eval);
        }
    }

    return best_moves;
}

OpeningEngine::OpeningEngine(const char *opening_path)
{
    FILE *opening_file = fopen(opening_path, "rb");

    Opening opening;
    while (fread(&opening, sizeof(opening), 1, opening_file) != 0)
    {
        this->openings_.push_back(opening);
    }

    std::sort(this->openings_.begin(), this->openings_.end(), &OpeningEngine::sort_fn);

    fclose(opening_file);
}

OpeningEngine::~OpeningEngine() {}

bool OpeningEngine::sort_fn(Opening const &a, Opening const &b)
{
    return a.game_cnt > b.game_cnt;
}

std::string OpeningEngine::next_move(Board *board, int move_cnt)
{
    std::string move_str = "";

    if (move_cnt >= CHESS_OPENING_MOVE_CNT)
    {
        return move_str;
    }

    for (auto opening : this->openings_)
    {
        if (board->compare_data(&opening.boards[(move_cnt - 1) * CHESS_BOARD_LEN]) == 0)
        {
            char *token = strtok(opening.move_strs, " ");
            for (int i = 0; i < move_cnt; i++)
            {
                token = strtok(NULL, " ");
            }

            std::string temp(token);
            move_str = temp;

            break;
        }
    }

    return move_str;
}

std::vector<PGNGame *> PGN::import(const char *path, long long file_size)
{
    FILE *file_ptr = fopen(path, "rb");

    char *buf = (char *)malloc(file_size);
    fread(buf, 1, file_size, file_ptr);

    fclose(file_ptr);

    std::vector<PGNGame *> games;
    PGNGame *game;

    for (int i = 0; i < file_size; i++)
    {
        if (i - 2 > 0 && buf[i - 2] == '\n' && buf[i - 1] == '1' && buf[i] == '.')
        {
            game = new PGNGame();

            i++;
            // At start of game (past "1.")).

            // It is possible that "1." could be "1. " (space after '.').
            while (i < file_size && (buf[i] == ' '))
            {
                i++;
            }

            // Read moves.

            while ((i + 1) < file_size && buf[i] != ' ' && buf[i + 1] != ' ' && (buf[i] != '1' && buf[i] != '0'))
            {

                // Turn x.

                // White move.
                std::string white_move_str;
                while (i < file_size && buf[i] != ' ' && buf[i] != '\n' && buf[i] != '\r')
                {
                    white_move_str += buf[i++];
                }
                game->move_strs.push_back(white_move_str);
                i++;

                while (i < file_size && (buf[i] == '\n' || buf[i] == '\r'))
                {
                    i++;
                }

                // It is possible that white made the last move.
                if ((i + 1) < file_size && buf[i] != ' ' && buf[i + 1] != ' ' && (buf[i] != '1' && buf[i] != '0'))
                {
                    // Black move.
                    std::string black_move_str;
                    while (i < file_size && buf[i] != ' ' && buf[i] != '\n' && buf[i] != '\r')
                    {
                        black_move_str += buf[i++];
                    }
                    game->move_strs.push_back(black_move_str);

                    while (i < file_size && (buf[i] == '\n' || buf[i] == '\r'))
                    {
                        i++;
                    }

                    // Go to next turn.
                    if ((i + 1) < file_size && buf[i + 1] != ' ' && buf[i + 1] != '/' && buf[i + 1] != '-' &&
                        ((i + 2) < file_size && buf[i + 2] != '/' && buf[i + 2] != '-'))
                    {
                        while (i < file_size && buf[i] != '.')
                        {
                            i++;
                        }
                        i++;

                        // buf[i] could be a space/newline.
                        while (i < file_size && (buf[i] == ' ' || buf[i] == '\n' || buf[i] == '\r'))
                        {
                            i++;
                        }
                    }
                }
            }

            // At end of game (right at/before 1-0 or 0-1 or 1/2-1/2 or *).
            // buf[i] could be a space/newline.
            while (i < file_size && (buf[i] == ' ' || buf[i] == '\n' || buf[i] == '\r'))
            {
                i++;
            }

            if (buf[i] == '0')
            {
                // Black win.
                game->lbl = -1;
            }
            else if (buf[i + 1] == '/' || buf[i] == '*' || buf[i + 1] == '*')
            {
                // Tie.
                game->lbl = 0;
            }
            else if (buf[i] == '1')
            {
                // White win.
                game->lbl = 1;
            }
            else
            {
                CHESS_THROW_ERROR("CHESS ERROR: invalid PGN result");
            }

            games.push_back(game);
        }

        // Ignore characters here...
    }

    free(buf);

    return games;
}

void PGN::export_openings(const char *pgn_path, long long pgn_file_size, const char *export_path)
{
    auto pgn_games = PGN::import(pgn_path, pgn_file_size);

    std::vector<Opening> openings;

    char board_data_buf[CHESS_BOARD_LEN * CHESS_OPENING_MOVE_CNT];
    std::string move_str_buf;

    for (auto pgn_game : pgn_games)
    {
        Board board;
        bool white = true;

        int game_move_cnt = 0;

        move_str_buf = "";

        for (auto move_str : pgn_game->move_strs)
        {
            auto move = board.change(move_str, white);
            white = !white;

            memcpy(&board_data_buf[game_move_cnt * CHESS_BOARD_LEN], board.get_data(), sizeof(char) * CHESS_BOARD_LEN);
            game_move_cnt++;

            move_str_buf += move_str;
            move_str_buf += " ";

            if (game_move_cnt >= CHESS_OPENING_MOVE_CNT)
            {
                bool match = false;
                for (int i = 0; i < openings.size(); i++)
                {
                    auto opening = &openings[i];

                    if (memcmp(opening->boards, board_data_buf, sizeof(board_data_buf)) == 0)
                    {
                        opening->game_cnt++;

                        match = true;
                        break;
                    }
                }

                if (!match)
                {
                    Opening opening;
                    memcpy(opening.boards, board_data_buf, sizeof(board_data_buf));
                    memset(opening.move_strs, 0, sizeof(move_str_buf));
                    memcpy(opening.move_strs, move_str_buf.c_str(), sizeof(opening.move_strs));

                    opening.game_cnt++;

                    openings.push_back(opening);
                }

                break;
            }
        }

        delete pgn_game;
    }

    FILE *openings_file = fopen(export_path, "wb");
    for (int i = 0; i < openings.size(); i++)
    {
        auto t = &openings[i];
        fwrite(t, sizeof(Opening), 1, openings_file);
    }
    fclose(openings_file);
}