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

int Board::get_square(int row, int col)
{
    return row * COL_CNT + col;
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

char Board::get_piece(int square)
{
    return this->data_[square];
}

bool Board::is_square_influenced(int square, bool by_white)
{
    for (int i = 0; i < BOARD_LEN; i++)
    {
        char piece = this->get_piece(i);

        if (by_white && Piece::is_white(piece))
        {
            auto moves = this->get_moves(i);
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
            auto moves = this->get_moves(i);
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

std::vector<Move> Board::get_moves(int square)
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

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) && Piece::is_black(this->get_piece(test_square)))
        {
            moves.push_back(Move{square, test_square});
        }

        test_row = row - 1;
        test_col = col + 1;
        test_square = Board::get_square(test_row, test_col);

        if (Board::is_row_valid(test_row) && Board::is_col_valid(test_col) && Piece::is_black(this->get_piece(test_square)))
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

        if (piece == WK && !this->castle_state.white_king_moved)
        {
            if (!this->castle_state.white_left_rook_moved)
            {
                if (this->get_piece(1) == MT && this->get_piece(2) == MT && this->get_piece(3) == MT)
                {
                    if (!this->is_square_influenced(1, false) && !this->is_square_influenced(2, false) &&
                        !this->is_square_influenced(3, false))
                    {
                        moves.push_back(Move{square, 2});
                    }
                }
            }

            if (!this->castle_state.white_right_rook_moved)
            {
                if (this->get_piece(5) == MT && this->get_piece(6) == MT)
                {
                    if (!this->is_square_influenced(5, false) && !this->is_square_influenced(6, false))
                    {
                        moves.push_back(Move{square, 6});
                    }
                }
            }
        }
        else if (piece == BK && !this->castle_state.black_king_moved)
        {
            if (!this->castle_state.black_left_rook_moved)
            {
                if (this->get_piece(57) == MT && this->get_piece(58) == MT && this->get_piece(59) == MT)
                {
                    if (!this->is_square_influenced(57, true) && !this->is_square_influenced(58, true) &&
                        !this->is_square_influenced(59, true))
                    {
                        moves.push_back(Move{square, 58});
                    }
                }
            }

            if (!this->castle_state.black_right_rook_moved)
            {
                if (this->get_piece(61) == MT && this->get_piece(62) == MT)
                {
                    if (!this->is_square_influenced(61, true) && !this->is_square_influenced(62, true))
                    {
                        moves.push_back(Move{square, 62});
                    }
                }
            }
        }
    }
    break;
    default:
        break;
    }

    return moves;
}

void Board::get_moves_parallel(int square)
{
    auto moves = this->get_moves(square);

    this->mutx.lock();
    this->all_moves.insert(all_moves.end(), moves.begin(), moves.end());
    this->mutx.unlock();
}

std::vector<Move> Board::get_all_moves(bool white)
{
    this->all_moves.clear();
    std::vector<std::thread> threads;

    if (white)
    {
        for (int i = 0; i < BOARD_LEN; i++)
        {
            if (Piece::is_white(this->get_piece(i)))
            {
                threads.push_back(std::thread(&Board::get_moves_parallel, this, i));
            }
        }
    }
    else
    {
        for (int i = 0; i < BOARD_LEN; i++)
        {
            if (Piece::is_black(this->get_piece(i)))
            {
                threads.push_back(std::thread(&Board::get_moves_parallel, this, i));
            }
        }
    }

    for (auto &th : threads)
    {
        th.join();
    }

    return this->all_moves;
}

void Board::change(Move move)
{
    char src_piece = this->data_[move.src_square];
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
        else if (src_col != dst_col)
        {
            this->data_[(dst_row - 1) * COL_CNT + dst_col] = MT;
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
        else if (src_col != dst_col)
        {
            this->data_[(dst_row + 1) * COL_CNT + dst_col] = MT;
        }
    }
    break;
    case WR:
    {
        if (src_col == 0)
        {
            this->castle_state.white_left_rook_moved = true;
        }
        else if (src_col == 7)
        {
            this->castle_state.white_right_rook_moved = true;
        }
    }
    break;
    case BR:
    {
        if (src_col == 0)
        {
            this->castle_state.black_left_rook_moved = true;
        }
        else if (src_col == 7)
        {
            this->castle_state.black_right_rook_moved = true;
        }
    }
    break;
    case WK:
    {
        // Look for castle.

        if (src_col - dst_col == -2)
        {
            this->data_[0] = MT;
            this->data_[3] = WR;
        }
        else if (src_col - dst_col == 2)
        {
            this->data_[7] = MT;
            this->data_[5] = WR;
        }

        this->castle_state.white_king_moved = true;
    }
    break;
    case BK:
    {
        // Look for castle.

        if (src_col - dst_col == -2)
        {
            this->data_[56] = MT;
            this->data_[59] = BR;
        }
        else if (src_col - dst_col == 2)
        {
            this->data_[63] = MT;
            this->data_[61] = BR;
        }

        this->castle_state.black_king_moved = true;
    }
    break;
    default:
        break;
    }

    this->data_[move.src_square] = MT;
    this->data_[move.dst_square] = dst_piece;
}