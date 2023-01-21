#include "position.h"

namespace schneizel
{
    void Position::init()
    {
        white_turn = true;

        // Pieces:
        {
            for (byte_t sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                this->pieces[sqnum] = PieceType::None;
            }

            this->pieces[8] = PieceType::WhitePawn;
            this->pieces[9] = PieceType::WhitePawn;
            this->pieces[10] = PieceType::WhitePawn;
            this->pieces[11] = PieceType::WhitePawn;
            this->pieces[12] = PieceType::WhitePawn;
            this->pieces[13] = PieceType::WhitePawn;
            this->pieces[14] = PieceType::WhitePawn;
            this->pieces[15] = PieceType::WhitePawn;

            this->pieces[1] = PieceType::WhiteKnight;
            this->pieces[6] = PieceType::WhiteKnight;
            this->pieces[2] = PieceType::WhiteBishop;
            this->pieces[5] = PieceType::WhiteBishop;
            this->pieces[0] = PieceType::WhiteRook;
            this->pieces[7] = PieceType::WhiteRook;
            this->pieces[3] = PieceType::WhiteQueen;
            this->pieces[4] = PieceType::WhiteKing;

            this->pieces[48] = PieceType::BlackPawn;
            this->pieces[49] = PieceType::BlackPawn;
            this->pieces[50] = PieceType::BlackPawn;
            this->pieces[51] = PieceType::BlackPawn;
            this->pieces[52] = PieceType::BlackPawn;
            this->pieces[53] = PieceType::BlackPawn;
            this->pieces[54] = PieceType::BlackPawn;
            this->pieces[55] = PieceType::BlackPawn;

            this->pieces[57] = PieceType::BlackKnight;
            this->pieces[62] = PieceType::BlackKnight;
            this->pieces[58] = PieceType::BlackBishop;
            this->pieces[61] = PieceType::BlackBishop;
            this->pieces[56] = PieceType::BlackRook;
            this->pieces[63] = PieceType::BlackRook;
            this->pieces[59] = PieceType::BlackQueen;
            this->pieces[60] = PieceType::BlackKing;
        }

        // Bitboards:
        {
            memset(this->piecebbs, 0, sizeof(this->piecebbs));
            this->whitebb = bitboards::EmptyBB;
            this->blackbb = bitboards::EmptyBB;

            for (byte_t sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                if (this->pieces[sqnum] != PieceType::None)
                    this->piecebbs[this->pieces[sqnum]] = bitboards::set_sqval(this->piecebbs[this->pieces[sqnum]], sqnum);
            }

            for (byte_t w = 0, b = PieceTypeCnt; w < PieceTypeCnt; w++, b++)
            {
                this->whitebb |= this->piecebbs[w];
                this->blackbb |= this->piecebbs[b];
            }
        }
    }

    void Position::pretty_print(Move *prev_move)
    {
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

        printf("\n");

        bool white_first = true;

        int foreground = 0;
        int background = 0;

        for (int row = 8 - 1; row >= 0; row--)
        {
            printf("%d  ", row + 1);
            printf("");

            for (int col = 0; col < 8; col++)
            {
                byte_t sqnum = row * 8 + col;
                PieceType piecetyp = this->pieces[sqnum];
                char piecetyp_char = get_piecetyp_char(piecetyp);

                if (piecetyp <= 5)
                {
                    foreground = 15;
                }
                else if (piecetyp >= 6 && piecetyp != PieceType::None)
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

                if (prev_move != nullptr && (sqnum == prev_move->src_sqnum || sqnum == prev_move->dst_sqnum))
                {
                    background = 4;
                }

                FlushConsoleInputBuffer(hConsole);
                SetConsoleTextAttribute(hConsole, foreground + background * 16);

                printf(" %c ", piecetyp_char);
            }

            white_first = !white_first;

            FlushConsoleInputBuffer(hConsole);
            SetConsoleTextAttribute(hConsole, 15);

            printf("\n");
        }

        FlushConsoleInputBuffer(hConsole);
        SetConsoleTextAttribute(hConsole, 15);

        printf("   ");
        printf(" a  b  c  d  e  f  g  h");

        printf("\n\n");
    }

    bitboard_t Position::get_allbb()
    {
        return this->whitebb | this->blackbb;
    }

    MoveList Position::get_move_list()
    {
        MoveList move_list;
        memset(&move_list, 0, sizeof(move_list));

        if (this->white_turn)
        {
            bitboard_t allbb = this->get_allbb();
            bitboard_t attackbb = bitboards::EmptyBB;

            for (byte_t src_sqnum = 0; src_sqnum < SquareCnt; src_sqnum++)
            {
                bitboard_t movebb;

                PieceType piecetyp = this->pieces[src_sqnum];
                bitboard_t piecebb = bitboards::get_sqbb(src_sqnum);

                switch (piecetyp)
                {
                case PieceType::WhitePawn:
                {
                    bitboard_t pawn_movebb = piecebb << 8;
                    bitboard_t pawn_movebb_2 = (piecebb & bitboards::Row2BB) << 16;

                    bitboard_t east_pawn_attackbb = (piecebb & ~bitboards::ColHBB) << 9;
                    bitboard_t west_pawn_attackbb = (piecebb & ~bitboards::ColABB) << 7;

                    pawn_movebb ^= allbb & pawn_movebb;
                    if (pawn_movebb != bitboards::EmptyBB)
                    {
                        pawn_movebb_2 ^= allbb & pawn_movebb_2;
                    }
                    else
                    {
                        pawn_movebb_2 = bitboards::EmptyBB;
                    }

                    east_pawn_attackbb &= this->blackbb;
                    west_pawn_attackbb &= this->blackbb;

                    attackbb |= east_pawn_attackbb;
                    attackbb |= west_pawn_attackbb;

                    movebb = pawn_movebb | pawn_movebb_2 | east_pawn_attackbb | west_pawn_attackbb;
                }
                break;
                case PieceType::WhiteKnight:
                {
                    movebb = bitboards::get_knight_movebb(src_sqnum) & ~this->whitebb;
                    attackbb |= movebb;
                }
                break;
                case PieceType::WhiteBishop:
                {
                    movebb = bitboards::get_bishop_movebb(src_sqnum, allbb) & ~this->whitebb;
                    attackbb |= movebb;
                }
                break;
                case PieceType::WhiteRook:
                {
                    movebb = bitboards::get_rook_movebb(src_sqnum, allbb) & ~this->whitebb;
                    attackbb |= movebb;
                }
                break;
                case PieceType::WhiteQueen:
                {
                    movebb = bitboards::get_queen_movebb(src_sqnum, allbb) & ~this->whitebb;
                    attackbb |= movebb;
                }
                break;
                case PieceType::WhiteKing:
                    movebb = bitboards::EmptyBB;
                    break;
                default:
                    movebb = bitboards::EmptyBB;
                    break;
                }

                // if (piecetyp <= 5)
                // {
                //     printf("%s\n", get_piecetyp_str(piecetyp));
                //     bitboards::print(movebb, src_sqnum);
                // }

                // for (byte_t dst_sqnum = 0; dst_sqnum < SquareCnt; dst_sqnum++)
                // {
                //     if (bitboards::get_sqval(movebb, dst_sqnum) == 1)
                //     {
                //         move_list.moves[move_list.move_cnt++] = Move{piecetyp, src_sqnum, dst_sqnum};
                //     }
                // }
                move_list.moves[move_list.move_cnt++] = Move{PieceType::WhitePawn, 8, 8};
            }
        }
        else
        {
            bitboard_t allbb = this->get_allbb();
            bitboard_t attackbb = bitboards::EmptyBB;

            for (byte_t src_sqnum = 0; src_sqnum < SquareCnt; src_sqnum++)
            {
                bitboard_t movebb;

                PieceType piecetyp = this->pieces[src_sqnum];
                bitboard_t piecebb = bitboards::get_sqbb(src_sqnum);

                switch (piecetyp)
                {
                case PieceType::BlackPawn:
                {
                    bitboard_t pawn_movebb = piecebb >> 8;
                    bitboard_t pawn_movebb_2 = (piecebb & bitboards::Row7BB) >> 16;

                    bitboard_t east_pawn_attackbb = (piecebb & ~bitboards::ColHBB) >> 7;
                    bitboard_t west_pawn_attackbb = (piecebb & ~bitboards::ColABB) >> 9;

                    pawn_movebb ^= allbb & pawn_movebb;
                    if (pawn_movebb != bitboards::EmptyBB)
                    {
                        pawn_movebb_2 ^= allbb & pawn_movebb_2;
                    }
                    else
                    {
                        pawn_movebb_2 = bitboards::EmptyBB;
                    }

                    east_pawn_attackbb &= this->whitebb;
                    west_pawn_attackbb &= this->whitebb;

                    attackbb |= east_pawn_attackbb;
                    attackbb |= west_pawn_attackbb;

                    movebb = pawn_movebb | pawn_movebb_2 | east_pawn_attackbb | west_pawn_attackbb;
                }
                break;
                case PieceType::BlackKnight:
                {
                    movebb = bitboards::get_knight_movebb(src_sqnum) & ~this->blackbb;
                    attackbb |= movebb;
                }
                break;
                case PieceType::BlackBishop:
                {
                    movebb = bitboards::get_bishop_movebb(src_sqnum, allbb) & ~this->blackbb;
                    attackbb |= movebb;
                }
                break;
                case PieceType::BlackRook:
                {
                    movebb = bitboards::get_rook_movebb(src_sqnum, allbb) & ~this->blackbb;
                    attackbb |= movebb;
                }
                break;
                case PieceType::BlackQueen:
                {
                    movebb = bitboards::get_queen_movebb(src_sqnum, allbb) & ~this->blackbb;
                    attackbb |= movebb;
                }
                break;
                case PieceType::BlackKing:
                    movebb = bitboards::EmptyBB;
                    break;
                default:
                    movebb = bitboards::EmptyBB;
                    break;
                }

                // if (piecetyp >= 6 && piecetyp != PieceType::None)
                // {
                //     printf("%s\n", get_piecetyp_str(piecetyp));
                //     bitboards::print(movebb, src_sqnum);
                // }

                // for (byte_t dst_sqnum = 0; dst_sqnum < SquareCnt; dst_sqnum++)
                // {
                //     if (bitboards::get_sqval(movebb, dst_sqnum) == 1)
                //     {
                //         move_list.moves[move_list.move_cnt++] = Move{piecetyp, src_sqnum, dst_sqnum};
                //     }
                // }
                move_list.moves[move_list.move_cnt++] = Move{PieceType::BlackPawn, 55, 55};
            }
        }

        return move_list;
    }

    void Position::make_move(Move move)
    {
        // TODO: promotion, castle, au passant

        PieceType src_piecetyp = this->pieces[move.src_sqnum];
        PieceType capture_piecetyp = this->pieces[move.dst_sqnum];

        bitboard_t movebb = bitboards::EmptyBB;
        movebb = bitboards::set_sqval(movebb, move.src_sqnum);
        movebb = bitboards::set_sqval(movebb, move.dst_sqnum);

        this->piecebbs[src_piecetyp] ^= movebb;

        if (capture_piecetyp != PieceType::None)
        {
            this->piecebbs[capture_piecetyp] = bitboards::clear_sqval(this->piecebbs[capture_piecetyp], move.dst_sqnum);
        }

        if (this->white_turn)
        {
            this->whitebb ^= movebb;
            this->blackbb &= ~movebb;
        }
        else
        {
            this->blackbb ^= movebb;
            this->whitebb &= ~movebb;
        }

        this->pieces[move.dst_sqnum] = src_piecetyp;
        this->pieces[move.src_sqnum] = PieceType::None;

        this->white_turn = !this->white_turn;
    }
}