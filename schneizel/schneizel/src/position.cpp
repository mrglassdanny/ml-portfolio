#include "position.h"

namespace schneizel
{
    void Position::init()
    {
        white_turn = true;

        // Pieces:
        {
            for (int sqnum = 0; sqnum < SquareCnt; sqnum++)
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
            this->whitebb = EmptyBB;
            this->blackbb = EmptyBB;

            for (int sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                if (this->pieces[sqnum] != PieceType::None)
                    this->piecebbs[this->pieces[sqnum]] = bitboards::set_sqval(this->piecebbs[this->pieces[sqnum]], sqnum);
            }

            for (int w = 0, b = PieceTypeCnt; w < PieceTypeCnt; w++, b++)
            {
                this->whitebb |= this->piecebbs[w];
                this->blackbb |= this->piecebbs[b];
            }
        }
    }

    bitboard_t Position::get_allbb()
    {
        return this->whitebb | this->blackbb;
    }

    void Position::get_moves()
    {
        MoveList move_list;

        if (this->white_turn)
        {
            bitboard_t allbb = this->get_allbb();

            bitboard_t pawn_movebb = this->piecebbs[PieceType::WhitePawn] << 8;

            bitboard_t row2_pawnbb = this->piecebbs[PieceType::WhitePawn] & bitboards::Row2BB;
            bitboard_t row2_pawn_movebb_2 = row2_pawnbb << 16;

            bitboard_t east_pawn_attackbb = (this->piecebbs[PieceType::WhitePawn] & ~bitboards::ColHBB) << 9;
            bitboard_t west_pawn_attackbb = (this->piecebbs[PieceType::WhitePawn] & ~bitboards::ColABB) << 7;

            pawn_movebb ^= allbb & pawn_movebb;
            row2_pawn_movebb_2 ^= allbb & pawn_movebb & row2_pawn_movebb_2;

            east_pawn_attackbb &= this->blackbb;
            west_pawn_attackbb &= this->blackbb;

            bitboards::print(pawn_movebb);
            bitboards::print(row2_pawn_movebb_2);
            bitboards::print(east_pawn_attackbb);
            bitboards::print(west_pawn_attackbb);

            bitboard_t movebb;

            for (int sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                PieceType typ = this->pieces[sqnum];

                switch (typ)
                {
                case PieceType::WhitePawn:
                    break;
                case PieceType::WhiteKnight:
                {
                    movebb = bitboards::get_knight_movebb(sqnum) & ~this->whitebb;
                    bitboards::print(movebb);
                }
                break;
                case PieceType::WhiteBishop:
                {
                    movebb = bitboards::get_bishop_movebb(sqnum, allbb) & ~this->whitebb;
                    bitboards::print(movebb);
                }
                break;
                case PieceType::WhiteRook:
                {
                    movebb = bitboards::get_rook_movebb(sqnum, allbb) & ~this->whitebb;
                    bitboards::print(movebb);
                }
                break;
                case PieceType::WhiteQueen:
                {
                    movebb = bitboards::get_queen_movebb(sqnum, allbb) & ~this->whitebb;
                    bitboards::print(movebb);
                }
                break;
                case PieceType::WhiteKing:
                    break;
                default:
                    break;
                }
            }
        }
        else
        {
        }
    }

    void Position::make_move(Move move)
    {
        // TODO: promotion, castle, au passant

        PieceType srctyp = this->pieces[move.src_sqnum];
        PieceType captyp = this->pieces[move.dst_sqnum];

        bitboard_t movebb = EmptyBB;
        movebb = bitboards::set_sqval(movebb, move.src_sqnum);
        movebb = bitboards::set_sqval(movebb, move.dst_sqnum);

        this->piecebbs[srctyp] ^= movebb;

        if (captyp != PieceType::None)
        {
            this->piecebbs[captyp] = bitboards::clear_sqval(this->piecebbs[captyp], move.dst_sqnum);
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

        this->pieces[move.dst_sqnum] = srctyp;
        this->pieces[move.src_sqnum] = PieceType::None;

        this->white_turn = !this->white_turn;
    }
}