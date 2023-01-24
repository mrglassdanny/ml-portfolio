#include "position.h"

namespace schneizel
{
    void Position::init()
    {
        white_turn = true;

        // Pieces:
        {
            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                this->pieces[sq] = PieceType::None;
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

            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                if (this->pieces[sq] != PieceType::None)
                    this->piecebbs[this->pieces[sq]] = bitboards::set_sqval(this->piecebbs[this->pieces[sq]], sq);
            }

            for (byte_t w = 0, b = PieceTypeCnt; w < PieceTypeCnt; w++, b++)
            {
                this->whitebb |= this->piecebbs[w];
                this->blackbb |= this->piecebbs[b];
            }

            this->allbb = this->whitebb | this->blackbb;
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
                square_t sq = row * 8 + col;
                PieceType piecetyp = this->pieces[sq];
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

                if (prev_move != nullptr && (sq == prev_move->src_sq || sq == prev_move->dst_sq))
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

    bitboard_t Position::get_whitebb()
    {
        return this->whitebb;
    }

    bitboard_t Position::get_blackbb()
    {
        return this->blackbb;
    }

    bitboard_t Position::get_allbb()
    {
        return this->allbb;
    }

    void Position::get_white_pawn_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t piecebb = bitboards::get_sqbb(src_sq);

        bitboard_t pawn_movebb = piecebb << 8;
        bitboard_t pawn_movebb_2 = (piecebb & bitboards::Row2BB) << 16;

        bitboard_t east_pawn_attackbb = (piecebb & ~bitboards::ColHBB) << 9;
        bitboard_t west_pawn_attackbb = (piecebb & ~bitboards::ColABB) << 7;

        pawn_movebb ^= this->allbb & pawn_movebb;
        if (pawn_movebb != bitboards::EmptyBB)
        {
            pawn_movebb_2 ^= this->allbb & pawn_movebb_2;
        }
        else
        {
            pawn_movebb_2 = bitboards::EmptyBB;
        }

        bitboard_t au_passant_blackbb = this->blackbb | bitboards::get_sqbb(this->au_passant_sq);

        east_pawn_attackbb &= au_passant_blackbb;
        west_pawn_attackbb &= au_passant_blackbb;

        this->white_attackbb |= east_pawn_attackbb;
        this->white_attackbb |= west_pawn_attackbb;

        bitboard_t movebb = pawn_movebb | pawn_movebb_2 | east_pawn_attackbb | west_pawn_attackbb;

        if ((piecebb & bitboards::Row7BB) != bitboards::EmptyBB)
        {
            while (movebb != bitboards::EmptyBB)
            {
                square_t dst_sq = pop_lsb(movebb);
                move_list->moves[move_list->move_cnt++] = Move(PieceType::WhitePawn, src_sq, dst_sq, PieceType::WhiteKnight);
                move_list->moves[move_list->move_cnt++] = Move(PieceType::WhitePawn, src_sq, dst_sq, PieceType::WhiteBishop);
                move_list->moves[move_list->move_cnt++] = Move(PieceType::WhitePawn, src_sq, dst_sq, PieceType::WhiteRook);
                move_list->moves[move_list->move_cnt++] = Move(PieceType::WhitePawn, src_sq, dst_sq, PieceType::WhiteQueen);
            }
        }
        else
        {
            while (movebb != bitboards::EmptyBB)
            {
                move_list->moves[move_list->move_cnt++] = Move{PieceType::WhitePawn, src_sq, pop_lsb(movebb)};
            }
        }
    }

    void Position::get_white_knight_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_knight_movebb(src_sq) & ~this->whitebb;
        this->white_attackbb |= movebb;

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteKnight, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_white_bishop_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_bishop_movebb(src_sq, this->allbb) & ~this->whitebb;
        this->white_attackbb |= movebb;

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteBishop, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_white_rook_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_rook_movebb(src_sq, this->allbb) & ~this->whitebb;
        this->white_attackbb |= movebb;

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteRook, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_white_queen_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_queen_movebb(src_sq, this->allbb) & ~this->whitebb;
        this->white_attackbb |= movebb;

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteQueen, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_white_king_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_king_movebb(src_sq) & ~this->whitebb;
        this->white_attackbb |= movebb;

        if (this->castle_rights.white_left)
        {
            // TODO
            if (bitboards::get_sqval(this->whitebb, 1) != 1 && bitboards::get_sqval(this->whitebb, 2) != 1 && bitboards::get_sqval(this->whitebb, 3) != 1)
            {
                movebb |= bitboards::get_sqbb(2);
            }
        }

        if (this->castle_rights.white_right)
        {
            // TODO
            if (bitboards::get_sqval(this->whitebb, 5) != 1 && bitboards::get_sqval(this->whitebb, 6) != 1)
            {
                movebb |= bitboards::get_sqbb(6);
            }
        }

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteKing, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_black_pawn_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t piecebb = bitboards::get_sqbb(src_sq);

        bitboard_t pawn_movebb = piecebb >> 8;
        bitboard_t pawn_movebb_2 = (piecebb & bitboards::Row7BB) >> 16;

        bitboard_t east_pawn_attackbb = (piecebb & ~bitboards::ColHBB) >> 7;
        bitboard_t west_pawn_attackbb = (piecebb & ~bitboards::ColABB) >> 9;

        pawn_movebb ^= this->allbb & pawn_movebb;
        if (pawn_movebb != bitboards::EmptyBB)
        {
            pawn_movebb_2 ^= this->allbb & pawn_movebb_2;
        }
        else
        {
            pawn_movebb_2 = bitboards::EmptyBB;
        }

        bitboard_t au_passant_whitebb = this->whitebb | bitboards::get_sqbb(this->au_passant_sq);

        east_pawn_attackbb &= au_passant_whitebb;
        west_pawn_attackbb &= au_passant_whitebb;

        this->black_attackbb |= east_pawn_attackbb;
        this->black_attackbb |= west_pawn_attackbb;

        bitboard_t movebb = pawn_movebb | pawn_movebb_2 | east_pawn_attackbb | west_pawn_attackbb;

        if ((piecebb & bitboards::Row2BB) != bitboards::EmptyBB)
        {
            while (movebb != bitboards::EmptyBB)
            {
                square_t dst_sq = pop_lsb(movebb);
                move_list->moves[move_list->move_cnt++] = Move(PieceType::BlackPawn, src_sq, dst_sq, PieceType::BlackKnight);
                move_list->moves[move_list->move_cnt++] = Move(PieceType::BlackPawn, src_sq, dst_sq, PieceType::BlackBishop);
                move_list->moves[move_list->move_cnt++] = Move(PieceType::BlackPawn, src_sq, dst_sq, PieceType::BlackRook);
                move_list->moves[move_list->move_cnt++] = Move(PieceType::BlackPawn, src_sq, dst_sq, PieceType::BlackQueen);
            }
        }
        else
        {
            while (movebb != bitboards::EmptyBB)
            {
                move_list->moves[move_list->move_cnt++] = Move{PieceType::BlackPawn, src_sq, pop_lsb(movebb)};
            }
        }
    }

    void Position::get_black_knight_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_knight_movebb(src_sq) & ~this->blackbb;
        this->black_attackbb |= movebb;

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteKnight, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_black_bishop_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_bishop_movebb(src_sq, this->allbb) & ~this->blackbb;
        this->black_attackbb |= movebb;

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteBishop, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_black_rook_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_rook_movebb(src_sq, this->allbb) & ~this->blackbb;
        this->black_attackbb |= movebb;

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteRook, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_black_queen_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_queen_movebb(src_sq, this->allbb) & ~this->blackbb;
        this->black_attackbb |= movebb;

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteQueen, src_sq, pop_lsb(movebb)};
        }
    }

    void Position::get_black_king_moves(square_t src_sq, MoveList *move_list)
    {
        bitboard_t movebb = bitboards::get_king_movebb(src_sq) & ~this->blackbb;
        this->black_attackbb |= movebb;

        if (this->castle_rights.black_left)
        {
            // TODO
            if (bitboards::get_sqval(this->blackbb, 57) != 1 && bitboards::get_sqval(this->blackbb, 58) != 1 && bitboards::get_sqval(this->blackbb, 59) != 1)
            {
                movebb |= bitboards::get_sqbb(58);
            }
        }

        if (this->castle_rights.black_right)
        {
            // TODO
            if (bitboards::get_sqval(this->blackbb, 61) != 1 && bitboards::get_sqval(this->blackbb, 62) != 1)
            {
                movebb |= bitboards::get_sqbb(62);
            }
        }

        while (movebb != bitboards::EmptyBB)
        {
            move_list->moves[move_list->move_cnt++] = Move{PieceType::WhiteKing, src_sq, pop_lsb(movebb)};
        }
    }

    MoveList Position::get_move_list()
    {
        MoveList move_list;
        move_list.move_cnt = 0;

        if (this->white_turn)
        {
            this->white_attackbb = bitboards::EmptyBB;

            // WhitePawn:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhitePawn];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_white_pawn_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // WhiteKnight:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteKnight];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_white_knight_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // WhiteBishop:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteBishop];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_white_bishop_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // WhiteRook:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteRook];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_white_rook_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // WhiteQueen:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteQueen];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_white_queen_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // WhiteKing:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteKing];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_white_king_moves(pop_lsb(piecebb), &move_list);
                }
            }
        }
        else
        {
            this->black_attackbb = bitboards::EmptyBB;

            // BlackPawn:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackPawn];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_black_pawn_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // BlackKnight:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackKnight];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_black_knight_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // BlackBishop:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackBishop];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_black_bishop_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // BlackRook:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackRook];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_black_rook_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // BlackQueen:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackQueen];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_black_queen_moves(pop_lsb(piecebb), &move_list);
                }
            }

            // BlackKing:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackKing];
                while (piecebb != bitboards::EmptyBB)
                {
                    this->get_black_king_moves(pop_lsb(piecebb), &move_list);
                }
            }
        }

        return move_list;
    }

    void Position::make_move(Move move)
    {
        PieceType src_piecetyp = this->pieces[move.src_sq];
        PieceType capture_piecetyp = this->pieces[move.dst_sq];

        // Castle rights:
        if (src_piecetyp == PieceType::WhiteRook)
        {
            if (move.src_sq == 0)
            {
                this->castle_rights.white_left = false;
            }
            else if (move.src_sq == 7)
            {
                this->castle_rights.white_right = false;
            }
        }
        else if (src_piecetyp == PieceType::BlackRook)
        {
            if (move.src_sq == 56)
            {
                this->castle_rights.black_left = false;
            }
            else if (move.src_sq == 63)
            {
                this->castle_rights.black_right = false;
            }
        }

        bitboard_t movebb = bitboards::EmptyBB;
        movebb = bitboards::set_sqval(movebb, move.src_sq);
        movebb = bitboards::set_sqval(movebb, move.dst_sq);

        // Castle:
        if (src_piecetyp == PieceType::WhiteKing)
        {
            if (move.src_sq == 4)
            {
                if (move.dst_sq == 2)
                {
                    this->pieces[3] = PieceType::WhiteRook;
                    this->pieces[0] = PieceType::None;
                    this->piecebbs[PieceType::WhiteRook] = bitboards::set_sqval(this->piecebbs[PieceType::WhiteRook], 3);
                    this->piecebbs[PieceType::WhiteRook] = bitboards::clear_sqval(this->piecebbs[PieceType::WhiteRook], 0);
                }
                else if (move.dst_sq == 6)
                {
                    this->pieces[5] = PieceType::WhiteRook;
                    this->pieces[7] = PieceType::None;
                    this->piecebbs[PieceType::WhiteRook] = bitboards::set_sqval(this->piecebbs[PieceType::WhiteRook], 5);
                    this->piecebbs[PieceType::WhiteRook] = bitboards::clear_sqval(this->piecebbs[PieceType::WhiteRook], 7);
                }
            }

            this->castle_rights.white_left = false;
            this->castle_rights.white_right = false;
        }
        else if (src_piecetyp == PieceType::BlackKing)
        {
            if (move.src_sq == 60)
            {
                if (move.dst_sq == 58)
                {
                    this->pieces[59] = PieceType::BlackRook;
                    this->pieces[56] = PieceType::None;
                    this->piecebbs[PieceType::BlackRook] = bitboards::set_sqval(this->piecebbs[PieceType::BlackRook], 59);
                    this->piecebbs[PieceType::BlackRook] = bitboards::clear_sqval(this->piecebbs[PieceType::BlackRook], 56);
                }
                else if (move.dst_sq == 62)
                {
                    this->pieces[61] = PieceType::BlackRook;
                    this->pieces[63] = PieceType::None;
                    this->piecebbs[PieceType::BlackRook] = bitboards::set_sqval(this->piecebbs[PieceType::BlackRook], 61);
                    this->piecebbs[PieceType::BlackRook] = bitboards::clear_sqval(this->piecebbs[PieceType::BlackRook], 63);
                }
            }

            this->castle_rights.black_left = false;
            this->castle_rights.black_right = false;
        }

        // Au passant:
        bool au_passant_opp = false;

        if (src_piecetyp == PieceType::WhitePawn)
        {
            if (move.dst_sq == this->au_passant_sq)
            {
                this->pieces[move.dst_sq - 8] = PieceType::None;
                this->piecebbs[PieceType::BlackPawn] = bitboards::clear_sqval(this->piecebbs[PieceType::BlackPawn], move.dst_sq - 8);
            }
            else if (move.dst_sq - move.src_sq > 9)
            {
                this->au_passant_sq = move.dst_sq - 8;
                au_passant_opp = true;
            }
        }
        else if (src_piecetyp == PieceType::BlackPawn)
        {
            if (move.dst_sq == this->au_passant_sq)
            {
                this->pieces[move.dst_sq + 8] = PieceType::None;
                this->piecebbs[PieceType::WhitePawn] = bitboards::clear_sqval(this->piecebbs[PieceType::BlackPawn], move.dst_sq + 8);
            }
            else if (move.src_sq - move.dst_sq > 9)
            {
                this->au_passant_sq = move.dst_sq + 8;
                au_passant_opp = true;
            }
        }

        if (!au_passant_opp)
        {
            this->au_passant_sq = this->white_turn ? 63 : 0;
        }

        // Regular vs promotion:
        if (move.promo_piecetyp != PieceType::None)
        {
            this->piecebbs[src_piecetyp] &= ~movebb;
            this->piecebbs[move.promo_piecetyp] = bitboards::set_sqval(this->piecebbs[move.promo_piecetyp], move.dst_sq);
        }
        else
        {
            this->piecebbs[src_piecetyp] ^= movebb;
        }

        // Capture:
        if (capture_piecetyp != PieceType::None)
        {
            this->piecebbs[capture_piecetyp] = bitboards::clear_sqval(this->piecebbs[capture_piecetyp], move.dst_sq);
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

        this->allbb = this->whitebb | this->blackbb;

        // Regular vs promotion:
        if (move.promo_piecetyp != PieceType::None)
        {
            this->pieces[move.dst_sq] = move.promo_piecetyp;
        }
        else
        {
            this->pieces[move.dst_sq] = src_piecetyp;
        }

        this->pieces[move.src_sq] = PieceType::None;

        this->white_turn = !this->white_turn;
    }
}