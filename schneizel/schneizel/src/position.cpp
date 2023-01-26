#include "position.h"

namespace schneizel
{
    CardinalDirection white_cardinal_direction_search_path[] = {CardinalDirection::North, CardinalDirection::East, CardinalDirection::West, CardinalDirection::South};
    DiagonalDirection white_diagonal_direction_search_path[] = {DiagonalDirection::Northeast, DiagonalDirection::Northwest, DiagonalDirection::Southeast, DiagonalDirection::Southwest};
    CardinalDirection black_cardinal_direction_search_path[] = {CardinalDirection::South, CardinalDirection::East, CardinalDirection::West, CardinalDirection::North};
    DiagonalDirection black_diagonal_direction_search_path[] = {DiagonalDirection::Southeast, DiagonalDirection::Southwest, DiagonalDirection::Northeast, DiagonalDirection::Northwest};

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

    PieceMoveList Position::get_white_pawn_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

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

        bitboard_t pin_filterbb = this->get_black_pin_filterbb(piecebb);

        piece_move_list.movebb = pawn_movebb | pawn_movebb_2 | east_pawn_attackbb | west_pawn_attackbb;
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = east_pawn_attackbb | west_pawn_attackbb;
        piece_move_list.attackbb &= pin_filterbb;

        return piece_move_list;
    }

    PieceMoveList Position::get_white_knight_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_knight_movebb(src_sq);

        bitboard_t pin_filterbb = this->get_black_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->whitebb;

        return piece_move_list;
    }

    PieceMoveList Position::get_white_bishop_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_bishop_movebb(src_sq, this->allbb);

        bitboard_t pin_filterbb = this->get_black_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->whitebb;

        // Pins:
        {
            bitboard_t king_attackbb = bitboards::get_bishop_movebb(src_sq, this->piecebbs[PieceType::BlackKing]);
            if ((king_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
            {
                bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                bitboard_t king_directionbb;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[white_diagonal_direction_search_path[i]];
                    if ((directionbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                    {
                        king_directionbb = directionbb;
                        break;
                    }
                }

                bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::BlackKing];
                bitboard_t pinbb = king_directionbb & all_but_kingbb;

                // Only a true pin if just 1 piece is in attack path.
                if (bitboards::popcount(pinbb ^ bitboards::get_sqbb(src_sq)) == 1)
                {
                    this->white_pins[src_sq].pinbb = pinbb;
                    this->white_pins[src_sq].king_directionbb = king_directionbb;
                    this->white_pins[src_sq].pinner_sq = src_sq;
                }
            }
        }

        return piece_move_list;
    }

    PieceMoveList Position::get_white_rook_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_rook_movebb(src_sq, this->allbb);

        bitboard_t pin_filterbb = this->get_black_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->whitebb;

        // Pins:
        {
            bitboard_t king_attackbb = bitboards::get_rook_movebb(src_sq, this->piecebbs[PieceType::BlackKing]);
            if ((king_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
            {
                bitboard_t *directionbbs = bitboards::get_rook_magic(src_sq)->directionbbs;
                bitboard_t king_directionbb;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[white_cardinal_direction_search_path[i]];
                    if ((directionbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                    {
                        king_directionbb = directionbb;
                        break;
                    }
                }

                bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::BlackKing];
                bitboard_t pinbb = king_directionbb & all_but_kingbb;

                // Only a true pin if just 1 piece is in attack path.
                if (bitboards::popcount(pinbb ^ bitboards::get_sqbb(src_sq)) == 1)
                {
                    this->white_pins[src_sq].pinbb = pinbb;
                    this->white_pins[src_sq].king_directionbb = king_directionbb;
                    this->white_pins[src_sq].pinner_sq = src_sq;
                }
            }
        }

        return piece_move_list;
    }

    PieceMoveList Position::get_white_queen_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_queen_movebb(src_sq, this->allbb);

        bitboard_t pin_filterbb = this->get_black_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->whitebb;

        // Pins:
        {
            bool king_in_diagonal_direction = false;
            // Diagonal:
            {
                bitboard_t king_attackbb = bitboards::get_bishop_movebb(src_sq, this->piecebbs[PieceType::BlackKing]);
                if ((king_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                {
                    bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                    bitboard_t king_directionbb;
                    for (int i = 0; i <= DirectionCnt; i++)
                    {
                        bitboard_t directionbb = directionbbs[white_diagonal_direction_search_path[i]];
                        if ((directionbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                        {
                            king_directionbb = directionbb;
                            break;
                        }
                    }

                    bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::BlackKing];
                    bitboard_t pinbb = king_directionbb & all_but_kingbb;

                    // Only a true pin if just 1 piece is in attack path.
                    if (bitboards::popcount(pinbb ^ bitboards::get_sqbb(src_sq)) == 1)
                    {
                        this->white_pins[src_sq].pinbb = pinbb;
                        this->white_pins[src_sq].king_directionbb = king_directionbb;
                        this->white_pins[src_sq].pinner_sq = src_sq;
                    }
                }
            }

            // Cardinal:
            if (!king_in_diagonal_direction)
            {
                bitboard_t king_attackbb = bitboards::get_rook_movebb(src_sq, this->piecebbs[PieceType::BlackKing]);
                if ((king_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                {
                    bitboard_t *directionbbs = bitboards::get_rook_magic(src_sq)->directionbbs;
                    bitboard_t king_directionbb;
                    for (int i = 0; i <= DirectionCnt; i++)
                    {
                        bitboard_t directionbb = directionbbs[white_cardinal_direction_search_path[i]];
                        if ((directionbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                        {
                            king_directionbb = directionbb;
                            break;
                        }
                    }

                    bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::BlackKing];
                    bitboard_t pinbb = king_directionbb & all_but_kingbb;

                    // Only a true pin if just 1 piece is in attack path.
                    if (bitboards::popcount(pinbb ^ bitboards::get_sqbb(src_sq)) == 1)
                    {
                        this->white_pins[src_sq].pinbb = pinbb;
                        this->white_pins[src_sq].king_directionbb = king_directionbb;
                        this->white_pins[src_sq].pinner_sq = src_sq;
                    }
                }
            }
        }

        return piece_move_list;
    }

    PieceMoveList Position::get_white_king_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_king_movebb(src_sq);
        piece_move_list.movebb &= this->black_attackbb;
        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->whitebb;

        if (this->castle_rights.white_left && bitboards::get_sqval(this->piecebbs[PieceType::WhiteRook], 0) == 1)
        {
            bitboard_t left_betweenbb = bitboards::get_white_castle_left_betweenbb();
            if ((left_betweenbb & this->allbb) == bitboards::EmptyBB && (left_betweenbb & this->black_attackbb) == bitboards::EmptyBB)
            {
                piece_move_list.movebb |= bitboards::get_sqbb(2);
            }
        }

        if (this->castle_rights.white_right && bitboards::get_sqval(this->piecebbs[PieceType::WhiteRook], 7) == 1)
        {
            bitboard_t right_betweenbb = bitboards::get_white_castle_right_betweenbb();
            if ((right_betweenbb & this->allbb) == bitboards::EmptyBB && (right_betweenbb & this->black_attackbb) == bitboards::EmptyBB)
            {
                piece_move_list.movebb |= bitboards::get_sqbb(6);
            }
        }

        return piece_move_list;
    }

    PieceMoveList Position::get_black_pawn_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

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

        bitboard_t pin_filterbb = this->get_white_pin_filterbb(piecebb);

        piece_move_list.movebb = pawn_movebb | pawn_movebb_2 | east_pawn_attackbb | west_pawn_attackbb;
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = east_pawn_attackbb | west_pawn_attackbb;
        piece_move_list.attackbb &= pin_filterbb;

        return piece_move_list;
    }

    PieceMoveList Position::get_black_knight_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_knight_movebb(src_sq);

        bitboard_t pin_filterbb = this->get_white_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->blackbb;

        return piece_move_list;
    }

    PieceMoveList Position::get_black_bishop_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_bishop_movebb(src_sq, this->allbb);

        bitboard_t pin_filterbb = this->get_white_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->blackbb;

        // Pins:
        {
            bitboard_t king_attackbb = bitboards::get_bishop_movebb(src_sq, this->piecebbs[PieceType::WhiteKing]);
            if ((king_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
            {
                bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                bitboard_t king_directionbb;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[black_diagonal_direction_search_path[i]];
                    if ((directionbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                    {
                        king_directionbb = directionbb;
                        break;
                    }
                }

                bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::WhiteKing];
                bitboard_t pinbb = king_directionbb & all_but_kingbb;

                // Only a true pin if just 1 piece is in attack path.
                if (bitboards::popcount(pinbb ^ bitboards::get_sqbb(src_sq)) == 1)
                {
                    this->black_pins[src_sq].pinbb = pinbb;
                    this->black_pins[src_sq].king_directionbb = king_directionbb;
                    this->black_pins[src_sq].pinner_sq = src_sq;
                }
            }
        }

        return piece_move_list;
    }

    PieceMoveList Position::get_black_rook_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_rook_movebb(src_sq, this->allbb);

        bitboard_t pin_filterbb = this->get_white_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->blackbb;

        // Pins:
        {
            bitboard_t king_attackbb = bitboards::get_rook_movebb(src_sq, this->piecebbs[PieceType::WhiteKing]);
            if ((king_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
            {
                bitboard_t *directionbbs = bitboards::get_rook_magic(src_sq)->directionbbs;
                bitboard_t king_directionbb;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[black_cardinal_direction_search_path[i]];
                    if ((directionbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                    {
                        king_directionbb = directionbb;
                        break;
                    }
                }

                bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::WhiteKing];
                bitboard_t pinbb = king_directionbb & all_but_kingbb;

                // Only a true pin if just 1 piece is in attack path.
                if (bitboards::popcount(pinbb ^ bitboards::get_sqbb(src_sq)) == 1)
                {
                    this->black_pins[src_sq].pinbb = pinbb;
                    this->black_pins[src_sq].king_directionbb = king_directionbb;
                    this->black_pins[src_sq].pinner_sq = src_sq;
                }
            }
        }

        return piece_move_list;
    }

    PieceMoveList Position::get_black_queen_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_queen_movebb(src_sq, this->allbb);

        bitboard_t pin_filterbb = this->get_white_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= pin_filterbb;

        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->blackbb;

        // Pins:
        {
            bool king_in_diagonal_direction = false;
            // Diagonal:
            {
                bitboard_t king_attackbb = bitboards::get_bishop_movebb(src_sq, this->piecebbs[PieceType::WhiteKing]);
                if ((king_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                {
                    bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                    bitboard_t king_directionbb;
                    for (int i = 0; i <= DirectionCnt; i++)
                    {
                        bitboard_t directionbb = directionbbs[black_diagonal_direction_search_path[i]];
                        if ((directionbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                        {
                            king_directionbb = directionbb;
                            break;
                        }
                    }

                    bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::WhiteKing];
                    bitboard_t pinbb = king_directionbb & all_but_kingbb;

                    // Only a true pin if just 1 piece is in attack path.
                    if (bitboards::popcount(pinbb ^ bitboards::get_sqbb(src_sq)) == 1)
                    {
                        this->black_pins[src_sq].pinbb = pinbb;
                        this->black_pins[src_sq].king_directionbb = king_directionbb;
                        this->black_pins[src_sq].pinner_sq = src_sq;
                    }
                }
            }

            // Cardinal:
            if (!king_in_diagonal_direction)
            {
                bitboard_t king_attackbb = bitboards::get_rook_movebb(src_sq, this->piecebbs[PieceType::WhiteKing]);
                if ((king_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                {
                    bitboard_t *directionbbs = bitboards::get_rook_magic(src_sq)->directionbbs;
                    bitboard_t king_directionbb;
                    for (int i = 0; i <= DirectionCnt; i++)
                    {
                        bitboard_t directionbb = directionbbs[black_cardinal_direction_search_path[i]];
                        if ((directionbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                        {
                            king_directionbb = directionbb;
                            break;
                        }
                    }

                    bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::WhiteKing];
                    bitboard_t pinbb = king_directionbb & all_but_kingbb;

                    // Only a true pin if just 1 piece is in attack path.
                    if (bitboards::popcount(pinbb ^ bitboards::get_sqbb(src_sq)) == 1)
                    {
                        this->black_pins[src_sq].pinbb = pinbb;
                        this->black_pins[src_sq].king_directionbb = king_directionbb;
                        this->black_pins[src_sq].pinner_sq = src_sq;
                    }
                }
            }
        }

        return piece_move_list;
    }

    PieceMoveList Position::get_black_king_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_king_movebb(src_sq);
        piece_move_list.movebb &= this->white_attackbb;
        piece_move_list.attackbb = piece_move_list.movebb;
        piece_move_list.movebb &= ~this->blackbb;

        if (this->castle_rights.black_left && bitboards::get_sqval(this->piecebbs[PieceType::BlackRook], 56) == 1)
        {
            bitboard_t left_betweenbb = bitboards::get_black_castle_left_betweenbb();
            if ((left_betweenbb & this->allbb) == bitboards::EmptyBB && (left_betweenbb & this->white_attackbb) == bitboards::EmptyBB)
            {
                piece_move_list.movebb |= bitboards::get_sqbb(58);
            }
        }

        if (this->castle_rights.black_right && bitboards::get_sqval(this->piecebbs[PieceType::BlackRook], 63) == 1)
        {
            bitboard_t right_betweenbb = bitboards::get_black_castle_right_betweenbb();
            if ((right_betweenbb & this->allbb) == bitboards::EmptyBB && (right_betweenbb & this->white_attackbb) == bitboards::EmptyBB)
            {
                piece_move_list.movebb |= bitboards::get_sqbb(62);
            }
        }

        return piece_move_list;
    }

    bool Position::is_in_check(bool white)
    {
        if (white)
        {
            this->get_move_list();
            if ((this->black_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
            {
                return true;
            }
        }
        else
        {
            this->get_move_list();
            if ((this->white_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
            {
                return true;
            }
        }

        return false;
    }

    bitboard_t Position::get_white_pin_filterbb(bitboard_t piecebb)
    {
        bitboard_t pin_filterbb = bitboards::FullBB;

        if ((piecebb & this->white_pinbb) != bitboards::EmptyBB)
        {
            for (int i = 0; i < this->white_pins_trimmed_cnt; i++)
            {
                if ((this->white_pins_trimmed[i]->king_directionbb & piecebb) != bitboards::EmptyBB)
                {
                    pin_filterbb = this->white_pins_trimmed[i]->king_directionbb;
                    this->pretty_print(nullptr);
                    bitboards::print(pin_filterbb);
                    bitboards::print(this->whitebb);
                    bitboards::print(this->white_pinbb);
                    bitboards::print(piecebb);
                    break;
                }
            }
        }

        return pin_filterbb;
    }

    bitboard_t Position::get_black_pin_filterbb(bitboard_t piecebb)
    {
        bitboard_t pin_filterbb = bitboards::FullBB;

        if ((piecebb & this->black_pinbb) != bitboards::EmptyBB)
        {
            for (int i = 0; i < this->black_pins_trimmed_cnt; i++)
            {
                if ((this->black_pins_trimmed[i]->king_directionbb & piecebb) != bitboards::EmptyBB)
                {
                    pin_filterbb = this->black_pins_trimmed[i]->king_directionbb;
                    this->pretty_print(nullptr);
                    bitboards::print(pin_filterbb);
                    bitboards::print(this->blackbb);
                    break;
                }
            }
        }

        return pin_filterbb;
    }

    MoveList Position::get_move_list()
    {
        MoveList move_list;
        move_list.move_cnt = 0;

        if (this->white_turn)
        {
            // Reset variables:
            this->white_attackbb = bitboards::EmptyBB;
            memset(this->white_attackbbs, 0, sizeof(this->white_attackbbs));
            this->white_pinbb = bitboards::EmptyBB;
            memset(this->white_pins, 0, sizeof(this->white_pins));
            memset(this->white_pins_trimmed, 0, sizeof(this->white_pins_trimmed));
            this->white_pins_trimmed_cnt = 0;

            // WhitePawn:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhitePawn];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_pawn_moves(src_sq);

                    if ((bitboards::get_sqbb(src_sq) & bitboards::Row7BB) != bitboards::EmptyBB)
                    {
                        while (piece_move_list.movebb != bitboards::EmptyBB)
                        {
                            square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                            move_list.moves[move_list.move_cnt++] = Move(PieceType::WhitePawn, src_sq, dst_sq, PieceType::WhiteKnight);
                            move_list.moves[move_list.move_cnt++] = Move(PieceType::WhitePawn, src_sq, dst_sq, PieceType::WhiteBishop);
                            move_list.moves[move_list.move_cnt++] = Move(PieceType::WhitePawn, src_sq, dst_sq, PieceType::WhiteRook);
                            move_list.moves[move_list.move_cnt++] = Move(PieceType::WhitePawn, src_sq, dst_sq, PieceType::WhiteQueen);
                        }
                    }
                    else
                    {
                        while (piece_move_list.movebb != bitboards::EmptyBB)
                        {
                            move_list.moves[move_list.move_cnt++] = Move{PieceType::WhitePawn, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                        }
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteKnight:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteKnight];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_knight_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::WhiteKnight, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteBishop:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteBishop];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_bishop_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::WhiteBishop, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteRook:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteRook];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_rook_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::WhiteRook, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteQueen:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteQueen];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_queen_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::WhiteQueen, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteKing:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::WhiteKing];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_king_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::WhiteKing, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }
        }
        else
        {
            // Reset variables:
            this->black_attackbb = bitboards::EmptyBB;
            memset(this->black_attackbbs, 0, sizeof(this->black_attackbbs));
            this->black_pinbb = bitboards::EmptyBB;
            memset(this->black_pins, 0, sizeof(this->black_pins));
            memset(this->black_pins_trimmed, 0, sizeof(this->black_pins_trimmed));
            this->black_pins_trimmed_cnt = 0;

            // BlackPawn:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackPawn];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_pawn_moves(src_sq);

                    if ((bitboards::get_sqbb(src_sq) & bitboards::Row2BB) != bitboards::EmptyBB)
                    {
                        while (piece_move_list.movebb != bitboards::EmptyBB)
                        {
                            square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                            move_list.moves[move_list.move_cnt++] = Move(PieceType::BlackPawn, src_sq, dst_sq, PieceType::BlackKnight);
                            move_list.moves[move_list.move_cnt++] = Move(PieceType::BlackPawn, src_sq, dst_sq, PieceType::BlackBishop);
                            move_list.moves[move_list.move_cnt++] = Move(PieceType::BlackPawn, src_sq, dst_sq, PieceType::BlackRook);
                            move_list.moves[move_list.move_cnt++] = Move(PieceType::BlackPawn, src_sq, dst_sq, PieceType::BlackQueen);
                        }
                    }
                    else
                    {
                        while (piece_move_list.movebb != bitboards::EmptyBB)
                        {
                            move_list.moves[move_list.move_cnt++] = Move{PieceType::BlackPawn, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                        }
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackKnight:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackKnight];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_knight_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::BlackKnight, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackBishop:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackBishop];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_bishop_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::BlackBishop, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackRook:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackRook];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_rook_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::BlackRook, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackQueen:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackQueen];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_queen_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::BlackQueen, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackKing:
            {
                bitboard_t piecebb = this->piecebbs[PieceType::BlackKing];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_king_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        move_list.moves[move_list.move_cnt++] = Move{PieceType::BlackKing, src_sq, bitboards::pop_lsb(piece_move_list.movebb)};
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }
        }

        return move_list;
    }

    void Position::make_move(Move move)
    {
        this->in_check = false;

        PieceType src_piecetyp = this->pieces[move.src_sq];
        PieceType capture_piecetyp = this->pieces[move.dst_sq];

        bitboard_t movebb = bitboards::EmptyBB;
        movebb = bitboards::set_sqval(movebb, move.src_sq);
        movebb = bitboards::set_sqval(movebb, move.dst_sq);

        // Castle:
        {
            if ((this->castle_rights.white_left || this->castle_rights.white_right) && src_piecetyp == PieceType::WhiteRook)
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
            else if ((this->castle_rights.black_left || this->castle_rights.black_right) && src_piecetyp == PieceType::BlackRook)
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
                        this->whitebb = bitboards::set_sqval(this->whitebb, 3);
                        this->whitebb = bitboards::clear_sqval(this->whitebb, 0);

                        PieceMoveList moved_rook_move_list = this->get_white_rook_moves(3);
                        this->white_attackbbs[0] = moved_rook_move_list.attackbb;
                    }
                    else if (move.dst_sq == 6)
                    {
                        this->pieces[5] = PieceType::WhiteRook;
                        this->pieces[7] = PieceType::None;
                        this->piecebbs[PieceType::WhiteRook] = bitboards::set_sqval(this->piecebbs[PieceType::WhiteRook], 5);
                        this->piecebbs[PieceType::WhiteRook] = bitboards::clear_sqval(this->piecebbs[PieceType::WhiteRook], 7);
                        this->whitebb = bitboards::set_sqval(this->whitebb, 5);
                        this->whitebb = bitboards::clear_sqval(this->whitebb, 7);

                        PieceMoveList moved_rook_move_list = this->get_white_rook_moves(5);
                        this->white_attackbbs[7] = moved_rook_move_list.attackbb;
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
                        this->blackbb = bitboards::set_sqval(this->blackbb, 59);
                        this->blackbb = bitboards::clear_sqval(this->blackbb, 56);

                        PieceMoveList moved_rook_move_list = this->get_black_rook_moves(59);
                        this->white_attackbbs[56] = moved_rook_move_list.attackbb;
                    }
                    else if (move.dst_sq == 62)
                    {
                        this->pieces[61] = PieceType::BlackRook;
                        this->pieces[63] = PieceType::None;
                        this->piecebbs[PieceType::BlackRook] = bitboards::set_sqval(this->piecebbs[PieceType::BlackRook], 61);
                        this->piecebbs[PieceType::BlackRook] = bitboards::clear_sqval(this->piecebbs[PieceType::BlackRook], 63);
                        this->blackbb = bitboards::set_sqval(this->blackbb, 61);
                        this->blackbb = bitboards::clear_sqval(this->blackbb, 63);

                        PieceMoveList moved_rook_move_list = this->get_black_rook_moves(61);
                        this->white_attackbbs[63] = moved_rook_move_list.attackbb;
                    }
                }

                this->castle_rights.black_left = false;
                this->castle_rights.black_right = false;
            }
        }

        // Au passant:
        {
            bool au_passant_opportunity = false;

            if (src_piecetyp == PieceType::WhitePawn)
            {
                square_t capture_au_passant_sq = move.dst_sq - 8;
                if (move.dst_sq == this->au_passant_sq)
                {
                    square_t au_passant_sq = move.dst_sq - 8;
                    this->pieces[au_passant_sq] = PieceType::None;
                    this->piecebbs[PieceType::BlackPawn] = bitboards::clear_sqval(this->piecebbs[PieceType::BlackPawn], au_passant_sq);
                    this->blackbb = bitboards::clear_sqval(this->blackbb, au_passant_sq);
                }
                else if (move.dst_sq - move.src_sq > 9)
                {
                    this->au_passant_sq = capture_au_passant_sq;
                    au_passant_opportunity = true;
                }
            }
            else if (src_piecetyp == PieceType::BlackPawn)
            {
                square_t capture_au_passant_sq = move.dst_sq + 8;
                if (move.dst_sq == this->au_passant_sq)
                {
                    this->pieces[capture_au_passant_sq] = PieceType::None;
                    this->piecebbs[PieceType::WhitePawn] = bitboards::clear_sqval(this->piecebbs[PieceType::WhitePawn], capture_au_passant_sq);
                    this->whitebb = bitboards::clear_sqval(this->whitebb, capture_au_passant_sq);
                }
                else if (move.src_sq - move.dst_sq > 9)
                {
                    this->au_passant_sq = capture_au_passant_sq;
                    au_passant_opportunity = true;
                }
            }

            if (!au_passant_opportunity)
            {
                this->au_passant_sq = this->white_turn ? 63 : 0;
            }
        }

        // Capture:
        if (capture_piecetyp != PieceType::None)
        {
            this->piecebbs[capture_piecetyp] = bitboards::clear_sqval(this->piecebbs[capture_piecetyp], move.dst_sq);

            // Need to invalid any captured pinners.
            memset(&this->white_pins[move.dst_sq], 0, sizeof(Pin));
            memset(&this->black_pins[move.dst_sq], 0, sizeof(Pin));
        }

        // Regular vs promotion:
        if (move.promo_piecetyp != PieceType::None)
        {
            this->piecebbs[src_piecetyp] = bitboards::clear_sqval(this->piecebbs[src_piecetyp], move.src_sq);
            this->piecebbs[move.promo_piecetyp] = bitboards::set_sqval(this->piecebbs[move.promo_piecetyp], move.dst_sq);

            this->pieces[move.dst_sq] = move.promo_piecetyp;
        }
        else
        {
            this->piecebbs[src_piecetyp] = bitboards::clear_sqval(this->piecebbs[src_piecetyp], move.src_sq);
            this->piecebbs[src_piecetyp] = bitboards::set_sqval(this->piecebbs[src_piecetyp], move.dst_sq);

            this->pieces[move.dst_sq] = src_piecetyp;
        }

        this->pieces[move.src_sq] = PieceType::None;

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

        // Update attacks for moved piece:
        PieceMoveList moved_piece_move_list;
        switch (this->pieces[move.dst_sq])
        {
        case PieceType::WhitePawn:
            moved_piece_move_list = this->get_white_pawn_moves(move.dst_sq);
            this->white_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::WhiteKnight:
            moved_piece_move_list = this->get_white_knight_moves(move.dst_sq);
            this->white_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::WhiteBishop:
            moved_piece_move_list = this->get_white_bishop_moves(move.dst_sq);
            this->white_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::WhiteRook:
            moved_piece_move_list = this->get_white_rook_moves(move.dst_sq);
            this->white_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::WhiteQueen:
            moved_piece_move_list = this->get_white_queen_moves(move.dst_sq);
            this->white_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::WhiteKing:
            moved_piece_move_list = this->get_white_king_moves(move.dst_sq);
            this->white_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::BlackPawn:
            moved_piece_move_list = this->get_black_pawn_moves(move.dst_sq);
            this->black_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::BlackKnight:
            moved_piece_move_list = this->get_black_knight_moves(move.dst_sq);
            this->black_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::BlackBishop:
            moved_piece_move_list = this->get_black_bishop_moves(move.dst_sq);
            this->black_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::BlackRook:
            moved_piece_move_list = this->get_black_rook_moves(move.dst_sq);
            this->black_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::BlackQueen:
            moved_piece_move_list = this->get_black_queen_moves(move.dst_sq);
            this->black_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        case PieceType::BlackKing:
            moved_piece_move_list = this->get_black_king_moves(move.dst_sq);
            this->black_attackbbs[move.src_sq] = moved_piece_move_list.attackbb;
            break;
        default:
            break;
        }

        // Update all attacks and see if checking king:
        if (this->white_turn)
        {
            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                this->white_attackbb |= this->white_attackbbs[sq];

                this->white_pinbb |= this->white_pins[sq].pinbb;
                if (this->white_pins[sq].pinbb != bitboards::EmptyBB)
                {
                    this->white_pins_trimmed[this->white_pins_trimmed_cnt++] = &this->white_pins[sq];
                }
            }

            if ((this->white_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
            {
                this->in_check = true;
            }
        }
        else
        {
            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                this->black_attackbb |= this->black_attackbbs[sq];

                this->black_pinbb |= this->black_pins[sq].pinbb;
                if (this->black_pins[sq].pinbb != bitboards::EmptyBB)
                {
                    this->black_pins_trimmed[this->black_pins_trimmed_cnt++] = &this->black_pins[sq];
                }
            }

            if ((this->black_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
            {
                this->in_check = true;
            }
        }

        this->white_turn = !this->white_turn;
    }
}