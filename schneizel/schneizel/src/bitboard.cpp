#include "bitboard.h"

bitboard_t Bitboard::get_knight_attacks(int sqnum)
{
    bitboard_t bb = Empty;

    int rownum = get_rownum_fr_sqnum(sqnum);
    int colnum = get_colnum_fr_sqnum(sqnum);

    int test_rownum;
    int test_colnum;

    test_rownum = rownum + 2;
    test_colnum = colnum + 1;
    if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
    {
        set_sq(bb, get_sqnum(test_rownum, test_colnum));
    }

    test_rownum = rownum + 2;
    test_colnum = colnum - 1;
    if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
    {
        set_sq(bb, get_sqnum(test_rownum, test_colnum));
    }

    test_rownum = rownum + 1;
    test_colnum = colnum + 2;
    if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
    {
        set_sq(bb, get_sqnum(test_rownum, test_colnum));
    }

    test_rownum = rownum + 1;
    test_colnum = colnum - 2;
    if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
    {
        set_sq(bb, get_sqnum(test_rownum, test_colnum));
    }

    test_rownum = rownum - 2;
    test_colnum = colnum + 1;
    if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
    {
        set_sq(bb, get_sqnum(test_rownum, test_colnum));
    }

    test_rownum = rownum - 2;
    test_colnum = colnum - 1;
    if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
    {
        set_sq(bb, get_sqnum(test_rownum, test_colnum));
    }

    test_rownum = rownum - 1;
    test_colnum = colnum + 2;
    if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
    {
        set_sq(bb, get_sqnum(test_rownum, test_colnum));
    }

    test_rownum = rownum - 1;
    test_colnum = colnum - 2;
    if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
    {
        set_sq(bb, get_sqnum(test_rownum, test_colnum));
    }

    return bb;
}

bitboard_t get_bishop_attacks(int sqnum, bitboard_t occupied)
{
    bitboard_t bb = Empty;

    int rownum = get_rownum_fr_sqnum(sqnum);
    int colnum = get_colnum_fr_sqnum(sqnum);

    int test_rownum;
    int test_colnum;
    int test_sqnum;

    // Northeast:
    test_rownum = rownum + 1;
    test_colnum = colnum + 1;
    test_sqnum = get_sqnum(test_rownum, test_colnum);
    while (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum) && !(occupied & get_sq(test_sqnum)))
    {
        set_sq(bb, test_sqnum);
        test_rownum++;
        test_colnum++;
        test_sqnum = get_sqnum(test_rownum, test_colnum);
    }

    // Northwest:
    test_rownum = rownum + 1;
    test_colnum = colnum - 1;
    test_sqnum = get_sqnum(test_rownum, test_colnum);
    while (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum) && !(occupied & get_sq(test_sqnum)))
    {
        set_sq(bb, test_sqnum);
        test_rownum++;
        test_colnum--;
        test_sqnum = get_sqnum(test_rownum, test_colnum);
    }

    // Southeast:
    test_rownum = rownum - 1;
    test_colnum = colnum + 1;
    test_sqnum = get_sqnum(test_rownum, test_colnum);
    while (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum) && !(occupied & get_sq(test_sqnum)))
    {
        set_sq(bb, test_sqnum);
        test_rownum--;
        test_colnum++;
        test_sqnum = get_sqnum(test_rownum, test_colnum);
    }

    // Southwest:
    test_rownum = rownum - 1;
    test_colnum = colnum - 1;
    test_sqnum = get_sqnum(test_rownum, test_colnum);
    while (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum) && !(occupied & get_sq(test_sqnum)))
    {
        set_sq(bb, test_sqnum);
        test_rownum--;
        test_colnum--;
        test_sqnum = get_sqnum(test_rownum, test_colnum);
    }

    return bb;
}

bitboard_t Bitboard::get_rook_attacks(int sqnum, bitboard_t occupied)
{
    bitboard_t bb = Empty;

    int rownum = get_rownum_fr_sqnum(sqnum);
    int colnum = get_colnum_fr_sqnum(sqnum);

    int test_rownum;
    int test_colnum;
    int test_sqnum;

    // North:
    test_rownum = rownum + 1;
    test_sqnum = get_sqnum(test_rownum, colnum);
    while (is_rownum_valid(test_rownum) && !(occupied & get_sq(test_sqnum)))
    {
        set_sq(bb, test_sqnum);
        test_rownum++;
        test_sqnum = get_sqnum(test_rownum, colnum);
    }

    // East:
    test_colnum = colnum + 1;
    test_sqnum = get_sqnum(rownum, test_colnum);
    while (is_colnum_valid(test_colnum) && !(occupied & get_sq(test_sqnum)))
    {
        set_sq(bb, test_sqnum);
        test_colnum++;
        test_sqnum = get_sqnum(rownum, test_colnum);
    }

    // West:
    test_colnum = colnum - 1;
    test_sqnum = get_sqnum(rownum, test_colnum);
    while (is_colnum_valid(test_colnum) && !(occupied & get_sq(test_sqnum)))
    {
        set_sq(bb, test_sqnum);
        test_colnum--;
        test_sqnum = get_sqnum(rownum, test_colnum);
    }

    // South:
    test_rownum = rownum - 1;
    test_sqnum = get_sqnum(test_rownum, colnum);
    while (is_rownum_valid(test_rownum) && !(occupied & get_sq(test_sqnum)))
    {
        set_sq(bb, test_sqnum);
        test_rownum--;
        test_sqnum = get_sqnum(test_rownum, colnum);
    }

    return bb;
}