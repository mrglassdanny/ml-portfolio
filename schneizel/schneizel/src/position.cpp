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
            memset(this->piece_bbs, 0, sizeof(this->piece_bbs));
            this->white_bb = Empty;
            this->black_bb = Empty;

            for (int sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                if (this->pieces[sqnum] != PieceType::None)
                    this->piece_bbs[this->pieces[sqnum]] = bitboards::set_sqval(this->piece_bbs[this->pieces[sqnum]], sqnum);
            }

            for (int w = 0, b = PieceTypeCnt; w < PieceTypeCnt; w++, b++)
            {
                this->white_bb |= this->piece_bbs[w];
                this->black_bb |= this->piece_bbs[b];
            }
        }
    }

    bitboard_t Position::get_all_bb()
    {
        return this->white_bb | this->black_bb;
    }

    void Position::get_moves()
    {
        MoveList move_list;

        if (this->white_turn)
        {
            bitboard_t all_bb = this->get_all_bb();

            bitboard_t orig_pawns = this->piece_bbs[PieceType::WhitePawn] & bitboards::Row2;
            bitboard_t orig_pawn_moves_1 = orig_pawns << 8;
            bitboard_t orig_pawn_moves_2 = orig_pawns << 16;

            bitboard_t east_pawn_attacks = (this->piece_bbs[PieceType::WhitePawn] & ~bitboards::ColH) << 9;
            bitboard_t west_pawn_attacks = (this->piece_bbs[PieceType::WhitePawn] & ~bitboards::ColA) << 7;

            bitboards::print(orig_pawns);
            bitboards::print(orig_pawn_moves_1);
            bitboards::print(orig_pawn_moves_2);
            bitboards::print(east_pawn_attacks);
            bitboards::print(west_pawn_attacks);

            orig_pawn_moves_1 ^= all_bb & orig_pawn_moves_1;
            orig_pawn_moves_2 ^= all_bb & orig_pawn_moves_1 & orig_pawn_moves_2;

            east_pawn_attacks &= this->black_bb;
            west_pawn_attacks &= this->black_bb;

            bitboards::print(orig_pawn_moves_1);
            bitboards::print(orig_pawn_moves_2);
            bitboards::print(east_pawn_attacks);
            bitboards::print(west_pawn_attacks);

            for (int sqnum = 0; sqnum < SquareCnt; sqnum++)
            {
                PieceType typ = this->pieces[sqnum];

                switch (typ)
                {
                case PieceType::WhitePawn:
                    break;
                case PieceType::WhiteKnight:
                {
                    bitboard_t knight_moves = bitboards::get_knight_moves(sqnum) & ~this->white_bb;
                    bitboards::print(knight_moves);
                }
                break;
                case PieceType::WhiteBishop:
                {
                    bitboard_t bishop_moves = bitboards::get_bishop_moves(sqnum, all_bb) & ~this->white_bb;
                    bitboards::print(bishop_moves);
                }
                break;
                case PieceType::WhiteRook:
                {
                    bitboard_t rook_moves = bitboards::get_rook_moves(sqnum, all_bb) & ~this->white_bb;
                    bitboards::print(rook_moves);
                }
                break;
                case PieceType::WhiteQueen:
                {
                    bitboard_t queen_moves = bitboards::get_queen_moves(sqnum, all_bb) & ~this->white_bb;
                    bitboards::print(queen_moves);
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

        bitboard_t move_bb = Empty;
        move_bb = bitboards::set_sqval(move_bb, move.src_sqnum);
        move_bb = bitboards::set_sqval(move_bb, move.dst_sqnum);

        this->piece_bbs[srctyp] ^= move_bb;

        if (captyp != PieceType::None)
        {
            this->piece_bbs[captyp] = bitboards::clear_sqval(this->piece_bbs[captyp], move.dst_sqnum);
        }

        if (this->white_turn)
        {
            this->white_bb ^= move_bb;
            this->black_bb &= ~move_bb;
        }
        else
        {
            this->black_bb ^= move_bb;
            this->white_bb &= ~move_bb;
        }

        this->pieces[move.dst_sqnum] = srctyp;
        this->pieces[move.src_sqnum] = PieceType::None;

        this->white_turn = !this->white_turn;
    }
}