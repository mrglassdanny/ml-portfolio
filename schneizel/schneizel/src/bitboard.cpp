#include "bitboard.h"

namespace schneizel
{
    namespace bitboards
    {
        bitboard_t knight_movebbs[SquareCnt];
        Magic bishop_magics[SquareCnt];
        Magic rook_magics[SquareCnt];

        class MagicPRNG
        {
        private:
            uint64_t s;
            uint64_t rand64()
            {
                s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
                return s * 2685821657736338717LL;
            }

        public:
            MagicPRNG(uint64_t seed) : s(seed) { assert(seed); }

            bitboard_t rand() { return (bitboard_t)rand64(); }

            bitboard_t sparse_rand()
            {
                return (bitboard_t)(rand64() & rand64() & rand64());
            }
        };

        int popcount(bitboard_t bb)
        {
            int cnt = 0;
            for (int i = 0; i < 64; i++)
            {
                cnt += get_sqval(bb, i);
            }
            return cnt;
        }

        Magic::~Magic()
        {
            free(this->movebbs);
        }

        unsigned Magic::get_movebb_index(bitboard_t blockerbb)
        {
            return unsigned(((blockerbb & this->maskbb) * this->keybb) >> this->shift);
        }

        bitboard_t init_knight_movebb(byte_t sqnum)
        {
            bitboard_t bb = EmptyBB;

            byte_t rownum = get_rownum_fr_sqnum(sqnum);
            byte_t colnum = get_colnum_fr_sqnum(sqnum);

            byte_t test_rownum;
            byte_t test_colnum;

            test_rownum = rownum + 2;
            test_colnum = colnum + 1;
            if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
            {
                bb = set_sqval(bb, get_sqnum(test_rownum, test_colnum));
            }

            test_rownum = rownum + 2;
            test_colnum = colnum - 1;
            if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
            {
                bb = set_sqval(bb, get_sqnum(test_rownum, test_colnum));
            }

            test_rownum = rownum + 1;
            test_colnum = colnum + 2;
            if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
            {
                bb = set_sqval(bb, get_sqnum(test_rownum, test_colnum));
            }

            test_rownum = rownum + 1;
            test_colnum = colnum - 2;
            if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
            {
                bb = set_sqval(bb, get_sqnum(test_rownum, test_colnum));
            }

            test_rownum = rownum - 2;
            test_colnum = colnum + 1;
            if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
            {
                bb = set_sqval(bb, get_sqnum(test_rownum, test_colnum));
            }

            test_rownum = rownum - 2;
            test_colnum = colnum - 1;
            if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
            {
                bb = set_sqval(bb, get_sqnum(test_rownum, test_colnum));
            }

            test_rownum = rownum - 1;
            test_colnum = colnum + 2;
            if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
            {
                bb = set_sqval(bb, get_sqnum(test_rownum, test_colnum));
            }

            test_rownum = rownum - 1;
            test_colnum = colnum - 2;
            if (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum))
            {
                bb = set_sqval(bb, get_sqnum(test_rownum, test_colnum));
            }

            return bb;
        }

        bitboard_t init_bishop_movebb(byte_t sqnum, bitboard_t blockersbb)
        {
            bitboard_t bb = EmptyBB;

            byte_t rownum = get_rownum_fr_sqnum(sqnum);
            byte_t colnum = get_colnum_fr_sqnum(sqnum);

            byte_t test_rownum;
            byte_t test_colnum;
            byte_t test_sqnum;

            // Northeast:
            test_rownum = rownum + 1;
            test_colnum = colnum + 1;
            test_sqnum = get_sqnum(test_rownum, test_colnum);
            while (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum) && !(blockersbb & get_sqbb(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_rownum++;
                test_colnum++;
                test_sqnum = get_sqnum(test_rownum, test_colnum);
            }

            // Northwest:
            test_rownum = rownum + 1;
            test_colnum = colnum - 1;
            test_sqnum = get_sqnum(test_rownum, test_colnum);
            while (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum) && !(blockersbb & get_sqbb(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_rownum++;
                test_colnum--;
                test_sqnum = get_sqnum(test_rownum, test_colnum);
            }

            // Southeast:
            test_rownum = rownum - 1;
            test_colnum = colnum + 1;
            test_sqnum = get_sqnum(test_rownum, test_colnum);
            while (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum) && !(blockersbb & get_sqbb(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_rownum--;
                test_colnum++;
                test_sqnum = get_sqnum(test_rownum, test_colnum);
            }

            // Southwest:
            test_rownum = rownum - 1;
            test_colnum = colnum - 1;
            test_sqnum = get_sqnum(test_rownum, test_colnum);
            while (is_rownum_valid(test_rownum) && is_colnum_valid(test_colnum) && !(blockersbb & get_sqbb(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_rownum--;
                test_colnum--;
                test_sqnum = get_sqnum(test_rownum, test_colnum);
            }

            return bb;
        }

        bitboard_t init_rook_movebb(byte_t sqnum, bitboard_t blockersbb)
        {
            bitboard_t bb = EmptyBB;

            byte_t rownum = get_rownum_fr_sqnum(sqnum);
            byte_t colnum = get_colnum_fr_sqnum(sqnum);

            byte_t test_rownum;
            byte_t test_colnum;
            byte_t test_sqnum;

            // North:
            test_rownum = rownum + 1;
            test_sqnum = get_sqnum(test_rownum, colnum);
            while (is_rownum_valid(test_rownum) && !(blockersbb & get_sqbb(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_rownum++;
                test_sqnum = get_sqnum(test_rownum, colnum);
            }

            // East:
            test_colnum = colnum + 1;
            test_sqnum = get_sqnum(rownum, test_colnum);
            while (is_colnum_valid(test_colnum) && !(blockersbb & get_sqbb(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_colnum++;
                test_sqnum = get_sqnum(rownum, test_colnum);
            }

            // West:
            test_colnum = colnum - 1;
            test_sqnum = get_sqnum(rownum, test_colnum);
            while (is_colnum_valid(test_colnum) && !(blockersbb & get_sqbb(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_colnum--;
                test_sqnum = get_sqnum(rownum, test_colnum);
            }

            // South:
            test_rownum = rownum - 1;
            test_sqnum = get_sqnum(test_rownum, colnum);
            while (is_rownum_valid(test_rownum) && !(blockersbb & get_sqbb(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_rownum--;
                test_sqnum = get_sqnum(test_rownum, colnum);
            }

            return bb;
        }

        void init_magics(bool bishop)
        {
            Magic *magics;
            bitboard_t (*movebb_fn)(byte_t sqnum, bitboard_t blockersbb);

            if (bishop)
            {
                magics = bishop_magics;
                movebb_fn = init_bishop_movebb;
            }
            else
            {
                magics = rook_magics;
                movebb_fn = init_rook_movebb;
            }

            int seeds[8] = {728, 10316, 55013, 32803, 12281, 15100, 16645, 255};

            bitboard_t blockerbbs[4096], movebbs[4096], edgebbs, blockerbb;
            int epoch[4096], epoch_cur = 0, move_cnt = 0;

            memset(epoch, 0, sizeof(epoch));

            for (byte_t sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                edgebbs = ((Row1BB | Row8BB) & ~get_rowbb_fr_sqnum(sqnum)) | ((ColABB | ColHBB) & ~get_colbb_fr_sqnum(sqnum));

                Magic *magic = &magics[sqnum];
                magic->maskbb = movebb_fn(sqnum, EmptyBB) & ~edgebbs;
                magic->shift = 64 - popcount(magic->maskbb);
                blockerbb = move_cnt = 0;

                do
                {
                    blockerbbs[move_cnt] = blockerbb;
                    movebbs[move_cnt] = movebb_fn(sqnum, blockerbb);
                    move_cnt++;
                    blockerbb = (blockerbb - magic->maskbb) & magic->maskbb;
                } while (blockerbb);

                magic->movebbs = (bitboard_t *)malloc(sizeof(bitboard_t) * move_cnt);
                memset(magic->movebbs, 0, sizeof(bitboard_t) * move_cnt);

                MagicPRNG magic_prng(seeds[get_rownum_fr_sqnum(sqnum)]);

                for (int i = 0; i < move_cnt;)
                {
                    for (magic->keybb = 0; popcount((magic->keybb * magic->maskbb) >> 56) < 6;)
                        magic->keybb = magic_prng.sparse_rand();

                    for (++epoch_cur, i = 0; i < move_cnt; ++i)
                    {
                        unsigned idx = magic->get_movebb_index(blockerbbs[i]);

                        if (epoch[idx] < epoch_cur)
                        {
                            epoch[idx] = epoch_cur;
                            magic->movebbs[idx] = movebbs[i];
                        }
                        else if (magic->movebbs[idx] != movebbs[i])
                            break;
                    }
                }
            }
        }

        void init()
        {
            // Knights:
            for (byte_t sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                knight_movebbs[sqnum] = init_knight_movebb(sqnum);
            }

            // Bishops:
            init_magics(true);

            // Rooks:
            init_magics(false);
        }

        void print(bitboard_t bb)
        {
            byte_t *bb_bytes = (byte_t *)&bb;
            for (int i = 8 - 1; i >= 0; i--)
            {
                printf("%d | ", i + 1);

                byte_t b = bb_bytes[i];
                for (int j = 0, k = 8; j < 8; j++, k--)
                {
                    byte_t b2 = (b >> j) & 1ULL;
                    printf("%u ", b2);
                }
                printf("\n");
            }
            printf("    ---------------\n");
            printf("    a b c d e f g h");
            printf("\n");
        }

        void print(bitboard_t bb, byte_t sqnum)
        {
            byte_t *bb_bytes = (byte_t *)&bb;
            for (int i = 8 - 1; i >= 0; i--)
            {
                printf("%d | ", i + 1);

                byte_t b = bb_bytes[i];
                for (int j = 0, k = 8; j < 8; j++, k--)
                {
                    if (i * 8 + j == sqnum)
                    {
                        printf("X ");
                    }
                    else
                    {
                        byte_t b2 = (b >> j) & 1ULL;
                        printf("%u ", b2);
                    }
                }
                printf("\n");
            }
            printf("    ---------------\n");
            printf("    a b c d e f g h");
            printf("\n");
        }

        bitboard_t get_knight_movebb(byte_t sqnum)
        {
            return knight_movebbs[sqnum];
        }

        bitboard_t get_bishop_movebb(byte_t sqnum, bitboard_t bodiesbb)
        {
            Magic *magic = &bishop_magics[sqnum];
            bitboard_t blockers = magic->maskbb & bodiesbb;
            unsigned idx = magic->get_movebb_index(blockers);
            return magic->movebbs[idx];
        }

        bitboard_t get_rook_movebb(byte_t sqnum, bitboard_t bodiesbb)
        {
            Magic *magic = &rook_magics[sqnum];
            bitboard_t blockers = magic->maskbb & bodiesbb;
            unsigned idx = magic->get_movebb_index(blockers);
            return magic->movebbs[idx];
        }

        bitboard_t get_queen_movebb(byte_t sqnum, bitboard_t bodiesbb)
        {
            return get_bishop_movebb(sqnum, bodiesbb) | get_rook_movebb(sqnum, bodiesbb);
        }
    }
}