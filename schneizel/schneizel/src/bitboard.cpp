#include "bitboard.h"

namespace schneizel
{
    namespace bitboards
    {
        bitboard_t knight_movebbs[SquareCnt];
        Magic bishop_magics[SquareCnt];
        Magic rook_magics[SquareCnt];
        bitboard_t king_movebbs[SquareCnt];

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
            MagicPRNG(uint64_t seed) : s(seed) {}

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

        bitboard_t init_knight_movebb(square_t sq)
        {
            bitboard_t bb = EmptyBB;

            row_t row = get_row_fr_sq(sq);
            col_t col = get_col_fr_sq(sq);

            row_t test_row;
            col_t test_col;

            test_row = row + 2;
            test_col = col + 1;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            test_row = row + 2;
            test_col = col - 1;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            test_row = row + 1;
            test_col = col + 2;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            test_row = row + 1;
            test_col = col - 2;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            test_row = row - 2;
            test_col = col + 1;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            test_row = row - 2;
            test_col = col - 1;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            test_row = row - 1;
            test_col = col + 2;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            test_row = row - 1;
            test_col = col - 2;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            return bb;
        }

        bitboard_t init_bishop_movebb(square_t sq, bitboard_t blockersbb)
        {
            bitboard_t bb = EmptyBB;

            row_t row = get_row_fr_sq(sq);
            col_t col = get_col_fr_sq(sq);

            row_t test_row;
            col_t test_col;
            square_t test_sq;

            // Northeast:
            test_row = row + 1;
            test_col = col + 1;
            test_sq = get_sq(test_row, test_col);
            while (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, test_sq);
                if ((blockersbb & get_sqbb(test_sq)) != EmptyBB)
                    break;

                test_row++;
                test_col++;
                test_sq = get_sq(test_row, test_col);
            }

            // Northwest:
            test_row = row + 1;
            test_col = col - 1;
            test_sq = get_sq(test_row, test_col);
            while (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, test_sq);
                if ((blockersbb & get_sqbb(test_sq)) != EmptyBB)
                    break;

                test_row++;
                test_col--;
                test_sq = get_sq(test_row, test_col);
            }

            // Southeast:
            test_row = row - 1;
            test_col = col + 1;
            test_sq = get_sq(test_row, test_col);
            while (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, test_sq);
                if ((blockersbb & get_sqbb(test_sq)) != EmptyBB)
                    break;

                test_row--;
                test_col++;
                test_sq = get_sq(test_row, test_col);
            }

            // Southwest:
            test_row = row - 1;
            test_col = col - 1;
            test_sq = get_sq(test_row, test_col);
            while (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, test_sq);
                if ((blockersbb & get_sqbb(test_sq)) != EmptyBB)
                    break;

                test_row--;
                test_col--;
                test_sq = get_sq(test_row, test_col);
            }

            return bb;
        }

        bitboard_t init_rook_movebb(square_t sq, bitboard_t blockersbb)
        {
            bitboard_t bb = EmptyBB;

            row_t row = get_row_fr_sq(sq);
            col_t col = get_col_fr_sq(sq);

            row_t test_row;
            col_t test_col;
            square_t test_sq;

            // North:
            test_row = row + 1;
            test_sq = get_sq(test_row, col);
            while (is_row_valid(test_row))
            {
                bb = set_sqval(bb, test_sq);
                if ((blockersbb & get_sqbb(test_sq)) != EmptyBB)
                    break;

                test_row++;
                test_sq = get_sq(test_row, col);
            }

            // East:
            test_col = col + 1;
            test_sq = get_sq(row, test_col);
            while (is_col_valid(test_col))
            {
                bb = set_sqval(bb, test_sq);
                if ((blockersbb & get_sqbb(test_sq)) != EmptyBB)
                    break;

                test_col++;
                test_sq = get_sq(row, test_col);
            }

            // West:
            test_col = col - 1;
            test_sq = get_sq(row, test_col);
            while (is_col_valid(test_col))
            {
                bb = set_sqval(bb, test_sq);
                if ((blockersbb & get_sqbb(test_sq)) != EmptyBB)
                    break;

                test_col--;
                test_sq = get_sq(row, test_col);
            }

            // South:
            test_row = row - 1;
            test_sq = get_sq(test_row, col);
            while (is_row_valid(test_row))
            {
                bb = set_sqval(bb, test_sq);
                if ((blockersbb & get_sqbb(test_sq)) != EmptyBB)
                    break;

                test_row--;
                test_sq = get_sq(test_row, col);
            }

            return bb;
        }

        bitboard_t init_king_movebb(square_t sq)
        {
            bitboard_t bb = EmptyBB;

            row_t row = get_row_fr_sq(sq);
            col_t col = get_col_fr_sq(sq);

            row_t test_row;
            col_t test_col;

            // North:
            test_row = row + 1;
            if (is_row_valid(test_row))
            {
                bb = set_sqval(bb, get_sq(test_row, col));
            }

            // East:
            test_col = col + 1;
            if (is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(row, test_col));
            }

            // West:
            test_col = col - 1;
            if (is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(row, test_col));
            }

            // South:
            test_row = row - 1;
            if (is_row_valid(test_row))
            {
                bb = set_sqval(bb, get_sq(test_row, col));
            }

            // Northeast:
            test_row = row + 1;
            test_col = col + 1;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            // Northwest:
            test_row = row + 1;
            test_col = col - 1;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            // Southeast:
            test_row = row - 1;
            test_col = col + 1;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            // Southwest:
            test_row = row - 1;
            test_col = col - 1;
            if (is_row_valid(test_row) && is_col_valid(test_col))
            {
                bb = set_sqval(bb, get_sq(test_row, test_col));
            }

            return bb;
        }

        void init_magics(bool bishop)
        {
            Magic *magics;
            bitboard_t (*movebb_fn)(square_t sq, bitboard_t blockersbb);

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

            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                edgebbs = ((Row1BB | Row8BB) & ~get_rowbb_fr_sq(sq)) | ((ColABB | ColHBB) & ~get_colbb_fr_sq(sq));

                Magic *magic = &magics[sq];
                magic->maskbb = movebb_fn(sq, EmptyBB) & ~edgebbs;
                magic->shift = 64 - popcount(magic->maskbb);
                blockerbb = move_cnt = 0;

                do
                {
                    blockerbbs[move_cnt] = blockerbb;
                    movebbs[move_cnt] = movebb_fn(sq, blockerbb);
                    move_cnt++;
                    blockerbb = (blockerbb - magic->maskbb) & magic->maskbb;
                } while (blockerbb);

                magic->movebbs = (bitboard_t *)malloc(sizeof(bitboard_t) * move_cnt);
                memset(magic->movebbs, 0, sizeof(bitboard_t) * move_cnt);

                MagicPRNG magic_prng(seeds[get_row_fr_sq(sq)]);

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
            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                knight_movebbs[sq] = init_knight_movebb(sq);
            }

            // Bishops:
            init_magics(true);

            // Rooks:
            init_magics(false);

            // Queens satisfied by bishops/rooks.

            // Kings:
            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                king_movebbs[sq] = init_king_movebb(sq);
            }
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

        void print(bitboard_t bb, square_t sq)
        {
            byte_t *bb_bytes = (byte_t *)&bb;
            for (int i = 8 - 1; i >= 0; i--)
            {
                printf("%d | ", i + 1);

                byte_t b = bb_bytes[i];
                for (int j = 0, k = 8; j < 8; j++, k--)
                {
                    if (i * 8 + j == sq)
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

        bitboard_t get_knight_movebb(square_t sq)
        {
            return knight_movebbs[sq];
        }

        bitboard_t get_bishop_movebb(square_t sq, bitboard_t bodiesbb)
        {
            Magic *magic = &bishop_magics[sq];
            bitboard_t blockersbb = magic->maskbb & bodiesbb;
            unsigned idx = magic->get_movebb_index(blockersbb);
            return magic->movebbs[idx];
        }

        bitboard_t get_rook_movebb(square_t sq, bitboard_t bodiesbb)
        {
            Magic *magic = &rook_magics[sq];
            bitboard_t blockersbb = magic->maskbb & bodiesbb;
            unsigned idx = magic->get_movebb_index(blockersbb);
            return magic->movebbs[idx];
        }

        bitboard_t get_queen_movebb(square_t sq, bitboard_t bodiesbb)
        {
            return get_bishop_movebb(sq, bodiesbb) | get_rook_movebb(sq, bodiesbb);
        }

        bitboard_t get_king_movebb(square_t sq)
        {
            return king_movebbs[sq];
        }
    }
}