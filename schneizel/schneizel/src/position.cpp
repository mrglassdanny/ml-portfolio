#include "position.h"

namespace schneizel
{
    void Position::init()
    {
        white_turn = true;

        // White pieces:
        {
            white_pawns = Row2;

            white_knights = Empty;
            white_knights = set_sqval(white_knights, 1);
            white_knights = set_sqval(white_knights, 6);

            white_bishops = Empty;
            white_bishops = set_sqval(white_bishops, 2);
            white_bishops = set_sqval(white_bishops, 5);

            white_rooks = Empty;
            white_rooks = set_sqval(white_rooks, 0);
            white_rooks = set_sqval(white_rooks, 7);

            white_queens = Empty;
            white_queens = set_sqval(white_queens, 3);

            white_king = Empty;
            white_king = set_sqval(white_king, 4);
        }

        // Black pieces:
        {
            black_pawns = Row7;

            black_knights = Empty;
            black_knights = set_sqval(black_knights, 57);
            black_knights = set_sqval(black_knights, 62);

            black_bishops = Empty;
            black_bishops = set_sqval(black_bishops, 58);
            black_bishops = set_sqval(black_bishops, 61);

            black_rooks = Empty;
            black_rooks = set_sqval(black_rooks, 56);
            black_rooks = set_sqval(black_rooks, 63);

            black_queens = Empty;
            black_queens = set_sqval(black_queens, 59);

            black_king = Empty;
            black_king = set_sqval(black_king, 60);
        }
    }

    bitboard_t Position::get_white_pieces()
    {
        return white_pawns | white_knights | white_bishops | white_rooks | white_queens | white_king;
    }

    bitboard_t Position::get_black_pieces()
    {
        return black_pawns | black_knights | black_bishops | black_rooks | black_queens | black_king;
    }

    bitboard_t Position::get_all_pieces()
    {
        return get_white_pieces() | get_black_pieces();
    }

    bitboard_t Position::get_friendly_pieces(bool white)
    {
        return white ? get_white_pieces() : get_black_pieces();
    }
}