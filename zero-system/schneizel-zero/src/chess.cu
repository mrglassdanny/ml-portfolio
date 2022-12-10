#include "chess.cuh"

using namespace chess;

char BOARD_START_STATE[CHESS_BOARD_LEN] =
    {
        WR, WN, WB, WQ, WK, WB, WN, WR,
        WP, WP, WP, WP, WP, WP, WP, WP,
        MT, MT, MT, MT, MT, MT, MT, MT,
        MT, MT, MT, MT, MT, MT, MT, MT,
        MT, MT, MT, MT, MT, MT, MT, MT,
        MT, MT, MT, MT, MT, MT, MT, MT,
        BP, BP, BP, BP, BP, BP, BP, BP,
        BR, BN, BB, BQ, BK, BB, BN, BR};

bool Piece::is_white(char piece)
{
    switch (piece)
    {
    case WP:
    case WN:
    case WB:
    case WR:
    case WQ:
    case WK:
        return true;
    default:
        return false;
    }
}

bool Piece::is_black(char piece)
{
    switch (piece)
    {
    case BP:
    case BN:
    case BB:
    case BR:
    case BQ:
    case BK:
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
    case WP:
        return " P ";
    case BP:
        return " p ";
    case WN:
        return " N ";
    case BN:
        return " n ";
    case WB:
        return " B ";
    case BB:
        return " b ";
    case WR:
        return " R ";
    case BR:
        return " r ";
    case WQ:
        return " Q ";
    case BQ:
        return " q ";
    case WK:
        return " K ";
    case BK:
        return " k ";
    default:
        return "   ";
    }
}

int Piece::get_value(char piece)
{
    switch (piece)
    {
    case WP:
        return 1;
    case BP:
        return -1;
    case WN:
        return 3;
    case BN:
        return -3;
    case WB:
        return 3;
    case BB:
        return -3;
    case WR:
        return 5;
    case BR:
        return -5;
    case WQ:
        return 9;
    case BQ:
        return -9;
    case WK:
        return 2;
    case BK:
        return -2;
    default:
        return 0;
    }
}

char Piece::get_pgn_piece(char piece)
{
    switch (piece)
    {
    case WN:
    case BN:
        return 'N';
    case WB:
    case BB:
        return 'B';
    case WR:
    case BR:
        return 'R';
    case WQ:
    case BQ:
        return 'Q';
    case WK:
    case BK:
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
            return WN;
        }
        else
        {
            return BN;
        }
    case 'B':
        if (white)
        {
            return WB;
        }
        else
        {
            return BB;
        }
    case 'R':
        if (white)
        {
            return WR;
        }
        else
        {
            return BR;
        }
    case 'Q':
        if (white)
        {
            return WQ;
        }
        else
        {
            return BQ;
        }
    case 'K':
        if (white)
        {
            return WK;
        }
        else
        {
            return BK;
        }
    default:
        // Pawn will be 'P' (optional).
        if (white)
        {
            return WP;
        }
        else
        {
            return BP;
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
    memcpy(this->data_, BOARD_START_STATE, sizeof(char) * CHESS_BOARD_LEN);

    memset(&this->check_state_, 0, sizeof(this->check_state_));
}

void Board::copy(Board *src)
{
    memcpy(this->data_, src->data_, sizeof(char) * CHESS_BOARD_LEN);
    this->castle_state_ = src->castle_state_;
    this->check_state_ = src->check_state_;
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

char *Board::get_data()
{
    return this->data_;
}

char Board::get_piece(int square)
{
    return this->data_[square];
}

int Board::get_king_square(bool white)
{
    int king_square = -1;
    if (white)
    {
        for (int i = 0; i < CHESS_BOARD_LEN; i++)
        {
            if (this->get_piece(i) == WK)
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
            if (this->get_piece(i) == BK)
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

std::vector<Move> Board::get_diagonal_moves(int square, char piece, int row, int col)
{
    std::vector<Move> moves;

    bool white = Piece::is_white(this->get_piece(square));

    int cnt;
    switch (piece)
    {
    case WB:
    case BB:
    case WQ:
    case BQ:
        cnt = 8;
        break;
    case WK:
    case BK:
        cnt = 2;
        break;
    default:
        return moves;
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
                if (test_piece == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

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
                else if (test_piece != MT)
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
                if (test_piece == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

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
                else if (test_piece != MT)
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
                if (test_piece == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

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
                else if (test_piece != MT)
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
                if (test_piece == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

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
                else if (test_piece != MT)
                {
                    break;
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
    case WR:
    case BR:
    case WQ:
    case BQ:
        cnt = 8;
        break;
    case WK:
    case BK:
        cnt = 2;
        break;
    default:
        return moves;
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
                if (test_piece == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

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
                else if (test_piece != MT)
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
                if (test_piece == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

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
                else if (test_piece != MT)
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
                if (test_piece == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

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
                else if (test_piece != MT)
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
                if (test_piece == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
                else
                {
                    if (!Piece::is_same_color(piece, test_piece))
                    {
                        moves.push_back(Move{square, test_square});
                    }

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
                else if (test_piece != MT)
                {
                    break;
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

    if (piece == MT)
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
    case WP:
    {
        test_row = row + 1;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && this->get_piece(test_square) == MT)
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

            if (this->get_piece(test_square) == MT)
            {
                test_row = row + 2;
                test_col = col;
                test_square = Board::get_square(test_row, test_col);

                if (this->get_piece(test_square) == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
            }
        }
    }
    break;
    case BP:
    {
        test_row = row - 1;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && this->get_piece(test_square) == MT)
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

            if (this->get_piece(test_square) == MT)
            {
                test_row = row - 2;
                test_col = col;
                test_square = Board::get_square(test_row, test_col);

                if (this->get_piece(test_square) == MT)
                {
                    moves.push_back(Move{square, test_square});
                }
            }
        }
    }
    break;
    case WN:
    case BN:
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
    case WB:
    case BB:
    {
        auto diagonal_moves = this->get_diagonal_moves(square, piece, row, col);
        moves.insert(moves.end(), diagonal_moves.begin(), diagonal_moves.end());
    }
    break;
    case WR:
    case BR:
    {
        auto straight_moves = this->get_straight_moves(square, piece, row, col);
        moves.insert(moves.end(), straight_moves.begin(), straight_moves.end());
    }
    break;
    case WQ:
    case BQ:
    {
        auto diagonal_moves = this->get_diagonal_moves(square, piece, row, col);
        moves.insert(moves.end(), diagonal_moves.begin(), diagonal_moves.end());

        auto straight_moves = this->get_straight_moves(square, piece, row, col);
        moves.insert(moves.end(), straight_moves.begin(), straight_moves.end());
    }
    break;
    case WK:
    case BK:
    {
        auto diagonal_moves = this->get_diagonal_moves(square, piece, row, col);
        moves.insert(moves.end(), diagonal_moves.begin(), diagonal_moves.end());

        auto straight_moves = this->get_straight_moves(square, piece, row, col);
        moves.insert(moves.end(), straight_moves.begin(), straight_moves.end());

        if (test_check)
        {
            if (piece == WK && !this->castle_state_.white_king_moved)
            {
                if (this->get_piece(0) == WR && !this->castle_state_.white_left_rook_moved)
                {
                    if (this->get_piece(1) == MT && this->get_piece(2) == MT && this->get_piece(3) == MT)
                    {
                        if (!this->is_square_under_attack(1, false) && !this->is_square_under_attack(2, false) &&
                            !this->is_square_under_attack(3, false))
                        {
                            moves.push_back(Move{square, 2});
                        }
                    }
                }

                if (this->get_piece(7) == WR && !this->castle_state_.white_right_rook_moved)
                {
                    if (this->get_piece(5) == MT && this->get_piece(6) == MT)
                    {
                        if (!this->is_square_under_attack(5, false) && !this->is_square_under_attack(6, false))
                        {
                            moves.push_back(Move{square, 6});
                        }
                    }
                }
            }
            else if (piece == BK && !this->castle_state_.black_king_moved)
            {
                if (this->get_piece(56) == BR && !this->castle_state_.black_left_rook_moved)
                {
                    if (this->get_piece(57) == MT && this->get_piece(58) == MT && this->get_piece(59) == MT)
                    {
                        if (!this->is_square_under_attack(57, true) && !this->is_square_under_attack(58, true) &&
                            !this->is_square_under_attack(59, true))
                        {
                            moves.push_back(Move{square, 58});
                        }
                    }
                }

                if (this->get_piece(63) == BR && !this->castle_state_.black_right_rook_moved)
                {
                    if (this->get_piece(61) == MT && this->get_piece(62) == MT)
                    {
                        if (!this->is_square_under_attack(61, true) && !this->is_square_under_attack(62, true))
                        {
                            moves.push_back(Move{square, 62});
                        }
                    }
                }
            }

            // Make sure king is not moving into check.
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
        memset(this->check_state_.black_king_pins, 0, sizeof(this->check_state_.black_king_pins));

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
        memset(this->check_state_.white_king_pins, 0, sizeof(this->check_state_.white_king_pins));

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

std::string Board::convert_move_to_move_str(Move move)
{
    std::string move_str;

    char piece = this->get_piece(move.src_square);
    char pgn_piece = Piece::get_pgn_piece(piece);
    int src_col = Board::get_col(move.src_square);
    int src_row = Board::get_row(move.src_square);
    int dst_col = Board::get_col(move.dst_square);
    int dst_row = Board::get_row(move.dst_square);

    if (piece == WK || piece == BK)
    {
        if ((src_col - dst_col) == -2)
        {
            move_str = "O-O";
            return move_str;
        }
        else if ((src_col - dst_col) == 2)
        {
            move_str = "O-O-O";
            return move_str;
        }
    }

    move_str += pgn_piece;

    move_str += Board::get_alpha_col(src_col);
    move_str += (char)(src_row + 1 + '0');

    move_str += Board::get_alpha_col(dst_col);
    move_str += (char)(dst_row + 1 + '0');

    // Check for pawn promotion.
    if ((piece == WP && dst_row == 7) || (piece == BP && dst_row == 0))
    {
        move_str += '=';

        if (move.promo_piece != MT)
        {
            move_str += move.promo_piece;
        }
        else
        {
            move_str += 'Q';
        }
    }

    return move_str;
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

void Board::change(Move move)
{
    if (move.src_square == CHESS_INVALID_SQUARE || move.dst_square == CHESS_INVALID_SQUARE)
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

    switch (src_piece)
    {
    case WP:
    {
        // Look for promotion and au passant.

        if (dst_row == 7)
        {
            if (move.promo_piece == MT)
            {
                dst_piece = WQ;
            }
            else
            {
                dst_piece = move.promo_piece;
            }
        }
        else if (dst_row == 5)
        {
            if (this->get_piece(move.dst_square) == MT)
            {
                int test_au_passant_square = Board::get_square(dst_row - 1, dst_col);
                if (this->get_piece(test_au_passant_square) == BP)
                {
                    this->data_[test_au_passant_square] = MT;
                }
            }
        }
    }
    break;
    case BP:
    {
        // Look for promotion and au passant.

        if (dst_row == 0)
        {
            if (move.promo_piece == MT)
            {
                dst_piece = BQ;
            }
            else
            {
                dst_piece = move.promo_piece;
            }
        }
        else if (dst_row == 2)
        {
            if (this->get_piece(move.dst_square) == MT)
            {
                int test_au_passant_square = Board::get_square(dst_row + 1, dst_col);
                if (this->get_piece(test_au_passant_square) == WP)
                {
                    this->data_[test_au_passant_square] = MT;
                }
            }
        }
    }
    break;
    case WR:
    {
        if (src_col == 0)
        {
            this->castle_state_.white_left_rook_moved = true;
        }
        else if (src_col == 7)
        {
            this->castle_state_.white_right_rook_moved = true;
        }
    }
    break;
    case BR:
    {
        if (src_col == 0)
        {
            this->castle_state_.black_left_rook_moved = true;
        }
        else if (src_col == 7)
        {
            this->castle_state_.black_right_rook_moved = true;
        }
    }
    break;
    case WK:
    {
        // Look for castle.

        if (src_col - dst_col == 2)
        {
            this->data_[0] = MT;
            this->data_[3] = WR;
        }
        else if (src_col - dst_col == -2)
        {
            this->data_[7] = MT;
            this->data_[5] = WR;
        }

        this->castle_state_.white_king_moved = true;
    }
    break;
    case BK:
    {
        // Look for castle.

        if (src_col - dst_col == 2)
        {
            this->data_[56] = MT;
            this->data_[59] = BR;
        }
        else if (src_col - dst_col == -2)
        {
            this->data_[63] = MT;
            this->data_[61] = BR;
        }

        this->castle_state_.black_king_moved = true;
    }
    break;
    default:
        break;
    }

    this->data_[move.src_square] = MT;
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
        if (this->check_state_.black_king_pins[move.src_square])
        {
            this->check_state_.black_checked = true;
        }
    }
    else
    {
        if (this->check_state_.white_king_pins[move.src_square])
        {
            this->check_state_.white_checked = true;
        }
    }
}

Move Board::change(std::string move_str, bool white)
{
    // Need to reset pins state.
    this->get_all_moves(white);

    char piece;
    char promo_piece = MT;

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

            if (this->get_piece(prev_square) == WP)
            {
                src_square = prev_square;
            }
            else if (this->get_piece(prev_square_2) == WP)
            {
                src_square = prev_square_2;
            }
        }
        else
        {
            int prev_square = Board::get_square(dst_row + 1, dst_col);
            int prev_square_2 = Board::get_square(dst_row + 2, dst_col);

            if (this->get_piece(prev_square) == BP)
            {
                src_square = prev_square;
            }
            else if (this->get_piece(prev_square_2) == BP)
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
                        src_square = i;
                        break;
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
                        src_square = i;
                        break;
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
                    piece = WP;
                }
                else
                {
                    piece = BP;
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
                if (piece == WQ || piece == BQ)
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

float Board::sim_minimax_sync(Simulation sim, bool white, int depth, int alpha, int beta)
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
        return (float)sim.board.evaluate_material();
    }

    if (!white)
    {
        float best_eval_val = CHESS_EVAL_MIN_VAL;
        auto sim_sims = sim.board.simulate_all(true);

        for (auto sim_sim : sim_sims)
        {
            float eval_val = Board::sim_minimax_sync(sim_sim, true, depth - 1, alpha, beta);

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
        float best_eval_val = CHESS_EVAL_MAX_VAL;
        auto sim_sims = sim.board.simulate_all(false);

        for (auto sim_sim : sim_sims)
        {
            float eval_val = Board::sim_minimax_sync(sim_sim, false, depth - 1, alpha, beta);

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

void Board::sim_minimax_async(Simulation sim, bool white, int depth, int alpha, int beta, Evaluation *evals)
{
    float eval_val = Board::sim_minimax_sync(sim, white, depth, alpha, beta);
    evals[sim.idx] = Evaluation{eval_val, sim.move, sim.board};
}

Move Board::change_minimax_async(bool white, int depth)
{
    auto sw = new zero::core::CpuStopWatch();
    sw->start();

    Evaluation evals[CHESS_BOARD_LEN];

    auto sims = this->simulate_all(white);

    float min = CHESS_EVAL_MIN_VAL;
    float max = CHESS_EVAL_MAX_VAL;

    float best_eval_val = white ? min : max;
    Move best_move;

    std::vector<std::thread> threads;

    for (auto sim : sims)
    {
        threads.push_back(std::thread(Board::sim_minimax_async, sim, white, depth, min, max, evals));
    }

    for (auto &th : threads)
    {
        th.join();
    }

    std::vector<Evaluation> ties;

    for (int i = 0; i < sims.size(); i++)
    {
        auto eval = evals[i];

        if ((white && eval.value > best_eval_val) || (!white && eval.value < best_eval_val))
        {
            best_eval_val = eval.value;
            best_move = eval.move;

            ties.clear();
        }
        else if (eval.value == best_eval_val)
        {
            ties.push_back(eval);
        }
    }

    if (ties.size() > 0)
    {
        int rand_idx = rand() % ties.size();
        best_move = ties[rand_idx].move;
    }

    this->change(best_move);

    sw->stop();
    sw->print_elapsed_seconds();
    delete sw;

    return best_move;
}

Move Board::change_minimax_async(bool white, int depth, zero::nn::Model *model)
{
    auto sw = new zero::core::CpuStopWatch();
    sw->start();

    Evaluation evals[CHESS_BOARD_LEN];

    auto sims = this->simulate_all(white);

    float min = CHESS_EVAL_MIN_VAL;
    float max = CHESS_EVAL_MAX_VAL;

    float best_eval_val = white ? min : max;
    Move best_move;

    std::vector<std::thread> threads;

    for (auto sim : sims)
    {
        threads.push_back(std::thread(Board::sim_minimax_async, sim, white, depth, min, max, evals));
    }

    for (auto &th : threads)
    {
        th.join();
    }

    std::vector<Evaluation> ties;

    for (int i = 0; i < sims.size(); i++)
    {
        auto eval = evals[i];

        if ((white && eval.value > best_eval_val) || (!white && eval.value < best_eval_val))
        {
            best_eval_val = eval.value;
            best_move = eval.move;

            ties.clear();
        }
        else if (eval.value == best_eval_val)
        {
            ties.push_back(eval);
        }
    }

    for (auto tie : ties)
    {
        auto x = zero::core::Tensor::zeros(false, model->input_shape());
        tie.board.one_hot_encode(x->data());
        x->to_cuda();
        auto p = model->forward(x);
        delete x;

        float eval_val = tie.value + p->get_val(0);
        delete p;

        if ((white && eval_val > best_eval_val) || (!white && eval_val < best_eval_val))
        {
            best_eval_val = tie.value;
            best_move = tie.move;
        }
    }

    this->change(best_move);

    sw->stop();
    sw->print_elapsed_seconds();
    delete sw;

    return best_move;
}

void Board::one_hot_encode(float *out)
{
    for (int c = 0; c < 6; c++)
    {
        for (int i = 0; i < CHESS_ROW_CNT; i++)
        {
            for (int j = 0; j < CHESS_COL_CNT; j++)
            {
                int out_idx = (c * CHESS_BOARD_LEN) + (i * CHESS_COL_CNT) + j;
                int square = (i * CHESS_COL_CNT) + j;

                switch (c)
                {
                case 0:
                    if (this->get_piece(square) == WP)
                    {
                        out[out_idx] = 1.0f;
                    }
                    else if (this->get_piece(square) == BP)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 1:
                    if (this->get_piece(square) == WN)
                    {
                        out[out_idx] = 1.0f;
                    }
                    else if (this->get_piece(square) == BN)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 2:
                    if (this->get_piece(square) == WB)
                    {
                        out[out_idx] = 1.0f;
                    }
                    else if (this->get_piece(square) == BB)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 3:
                    if (this->get_piece(square) == WR)
                    {
                        out[out_idx] = 1.0f;
                    }
                    else if (this->get_piece(square) == BR)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                case 4:
                    if (this->get_piece(square) == WQ)
                    {
                        out[out_idx] = 1.0f;
                    }
                    else if (this->get_piece(square) == BQ)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                default:
                    if (this->get_piece(square) == WK)
                    {
                        out[out_idx] = 1.0f;
                    }
                    else if (this->get_piece(square) == BK)
                    {
                        out[out_idx] = -1.0f;
                    }
                    break;
                }
            }
        }
    }
}

long long PGN::get_file_size(const char *file_name)
{
    HANDLE hFile = CreateFile((LPCSTR)file_name, GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE)
        return -1; // error condition, could call GetLastError to find out more

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size))
    {
        CloseHandle(hFile);
        return -1; // error condition, could call GetLastError to find out more
    }

    CloseHandle(hFile);
    return size.QuadPart;
}

std::vector<PGNGame *> PGN::import(const char *file_name)
{
    FILE *file_ptr = fopen(file_name, "rb");

    fseek(file_ptr, 0L, SEEK_END);
    long long file_size = PGN::get_file_size(file_name);
    rewind(file_ptr);

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

            // Read moves.

            while ((i + 1) < file_size && buf[i] != ' ' && buf[i + 1] != ' ')
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

                // It is possible that white made the last move.
                if ((i + 1) < file_size && buf[i] != ' ' && buf[i + 1] != ' ')
                {
                    // Black move.
                    std::string black_move_str;
                    while (i < file_size && buf[i] != ' ' && buf[i] != '\n' && buf[i] != '\r')
                    {
                        black_move_str += buf[i++];
                    }
                    game->move_strs.push_back(black_move_str);

                    // Go to next turn.
                    if ((i + 1) < file_size && buf[i + 1] != ' ')
                    {
                        while (i < file_size && buf[i] != '.')
                        {
                            i++;
                        }
                        i++;
                    }
                }
            }

            // At end of game (right before 1-0 or 0-1 or 1/2-1/2).
            // Should be spaces.
            i++;
            i++;
            if (buf[i] == '0')
            {
                // White loss.
                game->lbl = -1;
            }
            else
            {
                // buf[i] == 1;
                // This could mean tie; let's check next char for '/'.
                if (buf[i + 1] == '/')
                {
                    // Tie.
                    game->lbl = 0;
                }
                else
                {
                    // White win.
                    game->lbl = 1;
                }
            }

            games.push_back(game);
        }

        // Ignore characters here...
    }

    free(buf);

    return games;
}