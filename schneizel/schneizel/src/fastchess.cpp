#include "fastchess.h"

using namespace fastchess;

char BOARD_START_STATE[BOARD_LEN] =
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

char Piece::get_algnote_id(char piece)
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

Board::Board()
{
    this->reset();
}

Board::~Board() {}

int Board::get_row(int square)
{
    return square / COL_CNT;
}

int Board::get_col(int square)
{
    return square % COL_CNT;
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
    return row * COL_CNT + col;
}

bool Board::is_row_valid(int row)
{
    return row >= 0 && row < ROW_CNT;
}

bool Board::is_col_valid(int col)
{
    return col >= 0 && col < COL_CNT;
}

void Board::reset()
{
    memcpy(this->data_, BOARD_START_STATE, sizeof(char) * BOARD_LEN);
}

void Board::copy(Board *src)
{
    memcpy(this->data_, src->data_, sizeof(char) * BOARD_LEN);
    this->castle_state_ = src->castle_state_;
}

void Board::print()
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    printf("\n");

    bool white_first = true;

    int foreground = 0;
    int background = 0;

    for (int row = ROW_CNT - 1; row >= 0; row--)
    {
        printf("%d  ", row + 1);
        printf("");

        for (int col = 0; col < COL_CNT; col++)
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
    for (int col = 0; col < COL_CNT; col++)
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

    for (int row = ROW_CNT - 1; row >= 0; row--)
    {
        printf("%d  ", row + 1);
        printf("");

        for (int col = 0; col < COL_CNT; col++)
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
    for (int col = 0; col < COL_CNT; col++)
    {
        printf(" %c ", Board::get_alpha_col(col));
    }

    printf("\n\n");
}

char Board::get_piece(int square)
{
    return this->data_[square];
}

bool Board::is_square_under_attack(int square, bool by_white)
{
    for (int i = 0; i < BOARD_LEN; i++)
    {
        char piece = this->get_piece(i);

        if (by_white && Piece::is_white(piece))
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
        else if (!by_white && Piece::is_black(piece))
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

    return false;
}

bool Board::is_check(bool by_white)
{
    if (by_white)
    {
        int black_king_square = -1;
        for (int i = BOARD_LEN - 1; i >= 0; i--)
        {
            if (this->get_piece(i) == BK)
            {
                black_king_square = i;
                break;
            }
        }

        for (int i = 0; i < BOARD_LEN; i++)
        {
            switch (this->get_piece(i))
            {
            case WP:
            case WN:
            case WB:
            case WR:
            case WQ:
            case WK:
            {
                auto moves = this->get_moves(i, false);
                for (auto move : moves)
                {
                    if (move.dst_square == black_king_square)
                    {
                        return true;
                    }
                }
            }
            break;
            default:
                break;
            }
        }
    }
    else
    {
        int white_king_square = -1;
        for (int i = 0; i < BOARD_LEN; i++)
        {
            if (this->get_piece(i) == WK)
            {
                white_king_square = i;
                break;
            }
        }

        for (int i = 0; i < BOARD_LEN; i++)
        {
            switch (this->get_piece(i))
            {
            case BP:
            case BN:
            case BB:
            case BR:
            case BQ:
            case BK:
            {
                auto moves = this->get_moves(i, false);
                for (auto move : moves)
                {
                    if (move.dst_square == white_king_square)
                    {
                        return true;
                    }
                }
            }
            break;
            default:
                break;
            }
        }
    }

    return false;
}

bool Board::is_checkmate(bool by_white, bool assume_check)
{
    if (!assume_check)
    {
        if (!this->is_check(by_white))
        {
            return false;
        }
    }

    auto sims = this->simulate_all(!by_white);
    return sims.size() == 0;
}

std::vector<Move> Board::get_diagonal_moves(int square, char piece, int row, int col)
{
    std::vector<Move> moves;

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

    bool ne = false;
    bool sw = false;
    bool se = false;
    bool nw = false;

    for (int i = 1; i < cnt; i++)
    {
        test_row = row + i;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (!ne && Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

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

                ne = true;
            }
        }

        test_row = row - i;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (!sw && Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

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

                sw = true;
            }
        }

        test_row = row - i;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (!se && Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

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

                se = true;
            }
        }

        test_row = row + i;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (!nw && Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

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

                nw = true;
            }
        }
    }

    return moves;
}

std::vector<Move> Board::get_straight_moves(int square, char piece, int row, int col)
{
    std::vector<Move> moves;

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

    bool n = false;
    bool s = false;
    bool e = false;
    bool w = false;

    for (int i = 1; i < cnt; i++)
    {
        test_row = row + i;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (!n && Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

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

                n = true;
            }
        }

        test_row = row - i;
        test_col = col;
        test_square = Board::get_square(test_row, test_col);

        if (!s && Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

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

                s = true;
            }
        }

        test_row = row;
        test_col = col + i;
        test_square = Board::get_square(test_row, test_col);

        if (!e && Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

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

                e = true;
            }
        }

        test_row = row;
        test_col = col - i;
        test_square = Board::get_square(test_row, test_col);

        if (!w && Board::is_row_valid(test_row) && Board::is_col_valid(test_col))
        {
            char test_piece = this->get_piece(test_square);

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

                w = true;
            }
        }
    }

    return moves;
}

std::vector<Move> Board::get_moves(int square, bool test_check)
{
    std::vector<Move> moves;

    char piece = this->get_piece(square);

    if (piece == MT)
    {
        return moves;
    }

    int row = Board::get_row(square);
    int col = Board::get_col(square);

    bool white = Piece::is_white(piece);

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
                if (!this->castle_state_.white_left_rook_moved && this->get_piece(0) == WR)
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

                if (!this->castle_state_.white_right_rook_moved && this->get_piece(7) == WR)
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
                if (!this->castle_state_.black_left_rook_moved && this->get_piece(56) == BR)
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

                if (!this->castle_state_.black_right_rook_moved && this->get_piece(63) == BR)
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
        }
    }
    break;
    default:
        break;
    }

    if (test_check)
    {
        std::vector<Move> tested_moves;

        for (auto move : moves)
        {
            auto sim = this->simulate(move);

            if (!sim.board.is_check(!white))
            {
                tested_moves.push_back(move);
            }
        }

        moves = tested_moves;
    }

    return moves;
}

std::vector<Move> Board::get_all_moves(bool white)
{
    std::vector<Move> all_moves;

    if (white)
    {
        for (int i = 0; i < BOARD_LEN; i++)
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
        for (int i = 0; i < BOARD_LEN; i++)
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

std::string Board::convert_move_to_algnote_move(Move move)
{
    std::string algnote_move;

    char piece = this->data_[move.src_square];
    char piece_algnote_id = Piece::get_algnote_id(piece);
    int src_col = Board::get_col(move.src_square);
    int src_row = Board::get_row(move.src_square);
    int dst_col = Board::get_col(move.dst_square);
    int dst_row = Board::get_row(move.dst_square);

    // Check for castle.
    if (piece == WK || piece == BK)
    {
        if ((src_col - dst_col) == -2)
        {
            algnote_move = "O-O";
            return algnote_move;
        }
        else if ((src_col - dst_col) == 2)
        {
            algnote_move = "O-O-O";
            return algnote_move;
        }
    }

    // Example format going forward: piece id|src col|src row|dst col|dst row|promo ind|promo piece id
    // ^always 7 chars

    algnote_move += piece_algnote_id;

    algnote_move += Board::get_alpha_col(src_col);
    algnote_move += (char)(src_row + 1 + '0');

    algnote_move += Board::get_alpha_col(dst_col);
    algnote_move += (char)(dst_row + 1 + '0');

    // Check for pawn promotion.
    if ((piece == WP && dst_row == 7) || (piece == BP && dst_row == 0))
    {
        algnote_move += '=';
        algnote_move += 'Q';
    }

    return algnote_move;
}

void Board::change(Move move)
{
    char src_piece = this->get_piece(move.src_square);
    char dst_piece = src_piece;

    int src_row = Board::get_row(move.src_square);
    int src_col = Board::get_col(move.src_square);
    int dst_row = Board::get_row(move.dst_square);
    int dst_col = Board::get_col(move.dst_square);

    switch (src_piece)
    {
    case WP:
    {
        // Look for promotion and au passant.

        if (Board::get_row(move.dst_square) == 7)
        {
            dst_piece = WQ;
        }
    }
    break;
    case BP:
    {
        // Look for promotion and au passant.

        if (Board::get_row(move.dst_square) == 0)
        {
            dst_piece = BQ;
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
}

void Board::change_random(bool white)
{
    auto all_moves = this->get_all_moves(white);

    int rand_move_idx = rand() % all_moves.size();

    this->change(all_moves[rand_move_idx]);
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

    for (int i = 0; i < BOARD_LEN; i++)
    {
        mat_eval += Piece::get_value(this->get_piece(i));
    }

    return mat_eval;
}

float Board::sim_minimax_sync(Simulation sim, bool white, int depth, float alpha, float beta)
{
    if (sim.board.is_checkmate(!white, false))
    {
        if (white)
        {
            return EVAL_MAX_VAL;
        }
        else
        {
            return EVAL_MIN_VAL;
        }
    }

    if (depth == 0)
    {
        return (float)sim.board.evaluate_material();
    }

    if (!white)
    {
        float best_eval_val = EVAL_MIN_VAL;
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
        float best_eval_val = EVAL_MAX_VAL;
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

void Board::sim_minimax_async(Simulation sim, bool white, int depth, float alpha, float beta, Evaluation *evals)
{
    float eval_val = Board::sim_minimax_sync(sim, white, depth, alpha, beta);
    evals[sim.idx] = Evaluation{eval_val, sim.move};
}

Move Board::change_minimax_async(bool white, int depth)
{
    auto sw = new CpuStopWatch();
    sw->start();

    Evaluation evals[BOARD_LEN];

    auto sims = this->simulate_all(white);

    float min = EVAL_MIN_VAL;
    float max = EVAL_MAX_VAL;

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

        printf("MOVE: %s (%d->%d)\tEVAL: %f\n", this->convert_move_to_algnote_move(eval.move).c_str(),
               eval.move.src_square, eval.move.dst_square, eval.value);

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

    printf("TIES: %d\n", ties.size());
    if (ties.size() > 0)
    {
        int rand_idx = rand() % ties.size();
        best_move = ties[rand_idx].move;
    }

    printf("BEST MOVE: %s (%d->%d)\tEVAL: %f\n", this->convert_move_to_algnote_move(best_move).c_str(),
           best_move.src_square, best_move.dst_square, best_eval_val);

    this->change(best_move);

    sw->stop();
    sw->print_elapsed_seconds();
    delete sw;

    return best_move;
}

CpuStopWatch::CpuStopWatch()
{
    this->beg_ = 0;
    this->end_ = 0;
}

CpuStopWatch::~CpuStopWatch()
{
}

void CpuStopWatch::start()
{
    this->beg_ = clock();
    this->end_ = this->beg_;
}

void CpuStopWatch::stop()
{
    this->end_ = clock();
}

double CpuStopWatch::get_elapsed_seconds()
{
    return ((double)(this->end_ - this->beg_)) / CLOCKS_PER_SEC;
}

void CpuStopWatch::print_elapsed_seconds()
{
    printf("ELAPSED SECONDS: %f\n", this->get_elapsed_seconds());
}