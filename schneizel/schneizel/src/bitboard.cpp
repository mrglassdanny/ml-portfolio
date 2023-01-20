#include "bitboard.h"

namespace schneizel
{
    namespace bitboards
    {
        bitboard_t knight_attacks_[SquareCnt];
        Magic bishop_magics_[SquareCnt];
        Magic rook_magics_[SquareCnt];

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
            free(this->attacks);
        }

        unsigned Magic::get_attack_index(bitboard_t occupied)
        {
            return unsigned(((occupied & this->mask) * this->key) >> this->shift);
        }

        bitboard_t get_knight_attacks(int sqnum)
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
                bb = set_sqval(bb, test_sqnum);
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
                bb = set_sqval(bb, test_sqnum);
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
                bb = set_sqval(bb, test_sqnum);
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
                bb = set_sqval(bb, test_sqnum);
                test_rownum--;
                test_colnum--;
                test_sqnum = get_sqnum(test_rownum, test_colnum);
            }

            return bb;
        }

        bitboard_t get_rook_attacks(int sqnum, bitboard_t occupied)
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
                bb = set_sqval(bb, test_sqnum);
                test_rownum++;
                test_sqnum = get_sqnum(test_rownum, colnum);
            }

            // East:
            test_colnum = colnum + 1;
            test_sqnum = get_sqnum(rownum, test_colnum);
            while (is_colnum_valid(test_colnum) && !(occupied & get_sq(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_colnum++;
                test_sqnum = get_sqnum(rownum, test_colnum);
            }

            // West:
            test_colnum = colnum - 1;
            test_sqnum = get_sqnum(rownum, test_colnum);
            while (is_colnum_valid(test_colnum) && !(occupied & get_sq(test_sqnum)))
            {
                bb = set_sqval(bb, test_sqnum);
                test_colnum--;
                test_sqnum = get_sqnum(rownum, test_colnum);
            }

            // South:
            test_rownum = rownum - 1;
            test_sqnum = get_sqnum(test_rownum, colnum);
            while (is_rownum_valid(test_rownum) && !(occupied & get_sq(test_sqnum)))
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
            bitboard_t (*attacks_fn)(int sqnum, bitboard_t occupied);

            if (bishop)
            {
                magics = bishop_magics_;
                attacks_fn = get_bishop_attacks;
            }
            else
            {
                magics = rook_magics_;
                attacks_fn = get_rook_attacks;
            }

            int seeds[8] = {728, 10316, 55013, 32803, 12281, 15100, 16645, 255};

            bitboard_t occupancies[4096], attacks[4096], edges, occupied;
            int epoch[4096], epoch_cur = 0, attack_cnt = 0;

            memset(epoch, 0, sizeof(epoch));

            for (int sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                edges = ((Row1 | Row8) & ~get_row_fr_sqnum(sqnum)) | ((ColA | ColH) & ~get_col_fr_sqnum(sqnum));

                Magic *magic = &magics[sqnum];
                magic->mask = attacks_fn(sqnum, Empty) & ~edges;
                magic->shift = 64 - popcount(magic->mask);
                occupied = attack_cnt = 0;
                do
                {
                    occupancies[attack_cnt] = occupied;
                    attacks[attack_cnt] = attacks_fn(sqnum, occupied);
                    attack_cnt++;
                    occupied = (occupied - magic->mask) & magic->mask;
                } while (occupied);

                magic->attacks = (bitboard_t *)malloc(sizeof(bitboard_t) * attack_cnt);
                memset(magic->attacks, 0, sizeof(bitboard_t) * attack_cnt);

                MagicPRNG magic_prng(seeds[get_rownum_fr_sqnum(sqnum)]);

                for (int i = 0; i < attack_cnt;)
                {
                    for (magic->key = 0; popcount((magic->key * magic->mask) >> 56) < 6;)
                        magic->key = magic_prng.sparse_rand();

                    for (++epoch_cur, i = 0; i < attack_cnt; ++i)
                    {
                        unsigned idx = magic->get_attack_index(occupancies[i]);

                        if (epoch[idx] < epoch_cur)
                        {
                            epoch[idx] = epoch_cur;
                            magic->attacks[idx] = attacks[i];
                        }
                        else if (magic->attacks[idx] != attacks[i])
                            break;
                    }
                }
            }
        }

        void init()
        {
            // Knights:
            for (int sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                knight_attacks_[sqnum] = get_knight_attacks(sqnum);
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

        bitboard_t get_knight_moves(int sqnum)
        {
            return knight_attacks_[sqnum];
        }

        bitboard_t get_bishop_moves(int sqnum, bitboard_t occupied)
        {
            Magic *magic = &bishop_magics_[sqnum];
            bitboard_t blockers = magic->mask & occupied;
            int idx = magic->get_attack_index(blockers);
            return magic->attacks[idx];
        }

        bitboard_t get_rook_moves(int sqnum, bitboard_t occupied)
        {
            Magic *magic = &rook_magics_[sqnum];
            bitboard_t blockers = magic->mask & occupied;
            int idx = magic->get_attack_index(blockers);
            return magic->attacks[idx];
        }

        bitboard_t get_queen_moves(int sqnum, bitboard_t occupied)
        {
            return get_bishop_moves(sqnum, occupied) | get_rook_moves(sqnum, occupied);
        }
    }
}