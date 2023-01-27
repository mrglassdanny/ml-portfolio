#include "position.h"

namespace schneizel
{
    CardinalDirection white_cardinal_direction_search_path[] = {CardinalDirection::North, CardinalDirection::East, CardinalDirection::West, CardinalDirection::South};
    DiagonalDirection white_diagonal_direction_search_path[] = {DiagonalDirection::Northeast, DiagonalDirection::Northwest, DiagonalDirection::Southeast, DiagonalDirection::Southwest};
    CardinalDirection black_cardinal_direction_search_path[] = {CardinalDirection::South, CardinalDirection::East, CardinalDirection::West, CardinalDirection::North};
    DiagonalDirection black_diagonal_direction_search_path[] = {DiagonalDirection::Southeast, DiagonalDirection::Southwest, DiagonalDirection::Northeast, DiagonalDirection::Northwest};

    void Position::init()
    {
        memset(this, 0, sizeof(Position));

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

        // Other:
        {
            this->white_attackbb = bitboards::EmptyBB;
            this->white_king_attackbb = bitboards::EmptyBB;
            memset(this->white_attackbbs, 0, sizeof(this->white_attackbbs));
            this->white_pinbb = bitboards::EmptyBB;
            memset(this->white_pins, 0, sizeof(this->white_pins));
            memset(this->white_pins_trimmed, 0, sizeof(this->white_pins_trimmed));
            this->white_pins_trimmed_cnt = 0;

            this->black_attackbb = bitboards::EmptyBB;
            this->black_king_attackbb = bitboards::EmptyBB;
            memset(this->black_attackbbs, 0, sizeof(this->black_attackbbs));
            this->black_pinbb = bitboards::EmptyBB;
            memset(this->black_pins, 0, sizeof(this->black_pins));
            memset(this->black_pins_trimmed, 0, sizeof(this->black_pins_trimmed));
            this->black_pins_trimmed_cnt = 0;

            this->checker_attackbb = bitboards::FullBB;
            this->checker_sqbb = bitboards::EmptyBB;
            this->discovered_checker_attackbb = bitboards::FullBB;
            this->discovered_checker_sqbb = bitboards::EmptyBB;
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

        piece_move_list.attackbb = east_pawn_attackbb | west_pawn_attackbb;

        bitboard_t pawn_attack_movebb = this->blackbb | bitboards::get_sqbb(this->au_passant_sq);
        east_pawn_attackbb &= pawn_attack_movebb;
        west_pawn_attackbb &= pawn_attack_movebb;

        piece_move_list.movebb = pawn_movebb | pawn_movebb_2 | east_pawn_attackbb | west_pawn_attackbb;
        piece_move_list.movebb &= this->get_black_pin_filterbb(piecebb);
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);

        return piece_move_list;
    }

    PieceMoveList Position::get_white_knight_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_knight_movebb(src_sq);
        piece_move_list.attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= this->get_black_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        piece_move_list.movebb &= ~this->whitebb;

        return piece_move_list;
    }

    PieceMoveList Position::get_white_bishop_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_bishop_movebb(src_sq, this->allbb);
        piece_move_list.attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= this->get_black_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        piece_move_list.movebb &= ~this->whitebb;

        // Pins and king attacks:
        {
            bitboard_t king_attackbb = bitboards::get_bishop_movebb(src_sq, this->piecebbs[PieceType::BlackKing]);
            if ((king_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
            {
                bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[white_diagonal_direction_search_path[i]];
                    if ((directionbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                    {
                        piece_move_list.king_directionbb = directionbb & (king_attackbb | bitboards::get_sqbb(src_sq));
                        piece_move_list.no_block_king_directionbb = directionbb;
                        break;
                    }
                }

                bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::BlackKing];
                bitboard_t pinbb = piece_move_list.king_directionbb & all_but_kingbb;

                // Only a true pin if just 1 piece is in attack path.
                if (bitboards::popcount(pinbb & ~bitboards::get_sqbb(src_sq)) == 1)
                {
                    this->white_pins[src_sq].pinbb = pinbb;
                    this->white_pins[src_sq].king_directionbb = piece_move_list.king_directionbb;
                    this->white_pins[src_sq].no_block_king_directionbb = piece_move_list.no_block_king_directionbb;
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
        piece_move_list.attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= this->get_black_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        piece_move_list.movebb &= ~this->whitebb;

        // Pins and king attacks:
        {
            bitboard_t king_attackbb = bitboards::get_rook_movebb(src_sq, this->piecebbs[PieceType::BlackKing]);
            if ((king_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
            {
                bitboard_t *directionbbs = bitboards::get_rook_magic(src_sq)->directionbbs;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[white_cardinal_direction_search_path[i]];
                    if ((directionbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                    {
                        piece_move_list.king_directionbb = directionbb & (king_attackbb | bitboards::get_sqbb(src_sq));
                        piece_move_list.no_block_king_directionbb = directionbb;
                        break;
                    }
                }

                bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::BlackKing];
                bitboard_t pinbb = piece_move_list.king_directionbb & all_but_kingbb;

                // Only a true pin if just 1 piece is in attack path.
                if (bitboards::popcount(pinbb & ~bitboards::get_sqbb(src_sq)) == 1)
                {
                    this->white_pins[src_sq].pinbb = pinbb;
                    this->white_pins[src_sq].king_directionbb = piece_move_list.king_directionbb;
                    this->white_pins[src_sq].no_block_king_directionbb = piece_move_list.no_block_king_directionbb;
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
        piece_move_list.attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= this->get_black_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        piece_move_list.movebb &= ~this->whitebb;

        // Pins and king attacks:
        {
            bool king_in_diagonal_direction = false;
            // Diagonal:
            {
                bitboard_t king_attackbb = bitboards::get_bishop_movebb(src_sq, this->piecebbs[PieceType::BlackKing]);
                if ((king_attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                {
                    bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                    for (int i = 0; i <= DirectionCnt; i++)
                    {
                        bitboard_t directionbb = directionbbs[white_diagonal_direction_search_path[i]];
                        if ((directionbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                        {
                            piece_move_list.king_directionbb = directionbb & (king_attackbb | bitboards::get_sqbb(src_sq));
                            piece_move_list.no_block_king_directionbb = directionbb;
                            king_in_diagonal_direction = true;
                            break;
                        }
                    }

                    bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::BlackKing];
                    bitboard_t pinbb = piece_move_list.king_directionbb & all_but_kingbb;

                    // Only a true pin if just 1 piece is in attack path.
                    if (bitboards::popcount(pinbb & ~bitboards::get_sqbb(src_sq)) == 1)
                    {
                        this->white_pins[src_sq].pinbb = pinbb;
                        this->white_pins[src_sq].king_directionbb = piece_move_list.king_directionbb;
                        this->white_pins[src_sq].no_block_king_directionbb = piece_move_list.no_block_king_directionbb;
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
                    for (int i = 0; i <= DirectionCnt; i++)
                    {
                        bitboard_t directionbb = directionbbs[white_cardinal_direction_search_path[i]];
                        if ((directionbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
                        {
                            piece_move_list.king_directionbb = directionbb & (king_attackbb | bitboards::get_sqbb(src_sq));
                            piece_move_list.no_block_king_directionbb = directionbb;
                            break;
                        }
                    }

                    bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::BlackKing];
                    bitboard_t pinbb = piece_move_list.king_directionbb & all_but_kingbb;

                    // Only a true pin if just 1 piece is in attack path.
                    if (bitboards::popcount(pinbb & ~bitboards::get_sqbb(src_sq)) == 1)
                    {
                        this->white_pins[src_sq].pinbb = pinbb;
                        this->white_pins[src_sq].king_directionbb = piece_move_list.king_directionbb;
                        this->white_pins[src_sq].no_block_king_directionbb = piece_move_list.no_block_king_directionbb;
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
        piece_move_list.attackbb = bitboards::EmptyBB;
        this->white_king_attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= ~this->black_attackbb;
        piece_move_list.movebb &= ~this->black_king_attackbb;

        // Checker bitboards are both initialized to full.
        if (this->checker_attackbb != bitboards::FullBB)
        {
            piece_move_list.movebb &= (~this->checker_attackbb | this->checker_sqbb);
        }
        if (this->discovered_checker_attackbb != bitboards::FullBB)
        {
            piece_move_list.movebb &= (~this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        }

        piece_move_list.movebb &= ~this->whitebb;

        if ((this->checker_sqbb == bitboards::EmptyBB && this->discovered_checker_sqbb == bitboards::EmptyBB) &&
            this->castle_rights.white_left && bitboards::get_sqval(this->piecebbs[PieceType::WhiteRook], 0) == 1)
        {
            bitboard_t left_betweenbb = bitboards::get_white_castle_left_betweenbb();
            if ((left_betweenbb & this->allbb) == bitboards::EmptyBB &&
                (left_betweenbb & this->black_attackbb) == bitboards::EmptyBB &&
                (left_betweenbb & this->black_king_attackbb) == bitboards::EmptyBB)
            {
                piece_move_list.movebb |= bitboards::get_sqbb(2);
            }
        }

        if ((this->checker_sqbb == bitboards::EmptyBB && this->discovered_checker_sqbb == bitboards::EmptyBB) &&
            this->castle_rights.white_right && bitboards::get_sqval(this->piecebbs[PieceType::WhiteRook], 7) == 1)
        {
            bitboard_t right_betweenbb = bitboards::get_white_castle_right_betweenbb();
            if ((right_betweenbb & this->allbb) == bitboards::EmptyBB &&
                (right_betweenbb & this->black_attackbb) == bitboards::EmptyBB &&
                (right_betweenbb & this->black_king_attackbb) == bitboards::EmptyBB)
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

        piece_move_list.attackbb = east_pawn_attackbb | west_pawn_attackbb;

        bitboard_t pawn_attack_movebb = this->whitebb | bitboards::get_sqbb(this->au_passant_sq);
        east_pawn_attackbb &= pawn_attack_movebb;
        west_pawn_attackbb &= pawn_attack_movebb;

        piece_move_list.movebb = pawn_movebb | pawn_movebb_2 | east_pawn_attackbb | west_pawn_attackbb;
        piece_move_list.movebb &= this->get_white_pin_filterbb(piecebb);
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);

        return piece_move_list;
    }

    PieceMoveList Position::get_black_knight_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_knight_movebb(src_sq);
        piece_move_list.attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= this->get_white_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        piece_move_list.movebb &= ~this->blackbb;

        return piece_move_list;
    }

    PieceMoveList Position::get_black_bishop_moves(square_t src_sq)
    {
        PieceMoveList piece_move_list;

        piece_move_list.movebb = bitboards::get_bishop_movebb(src_sq, this->allbb);
        piece_move_list.attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= this->get_white_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        piece_move_list.movebb &= ~this->blackbb;

        // Pins and king attacks:
        {
            bitboard_t king_attackbb = bitboards::get_bishop_movebb(src_sq, this->piecebbs[PieceType::WhiteKing]);
            if ((king_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
            {
                bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[black_diagonal_direction_search_path[i]];
                    if ((directionbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                    {
                        piece_move_list.king_directionbb = directionbb & (king_attackbb | bitboards::get_sqbb(src_sq));
                        piece_move_list.no_block_king_directionbb = directionbb;
                        break;
                    }
                }

                bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::WhiteKing];
                bitboard_t pinbb = piece_move_list.king_directionbb & all_but_kingbb;

                // Only a true pin if just 1 piece is in attack path.
                if (bitboards::popcount(pinbb & ~bitboards::get_sqbb(src_sq)) == 1)
                {
                    this->black_pins[src_sq].pinbb = pinbb;
                    this->black_pins[src_sq].king_directionbb = piece_move_list.king_directionbb;
                    this->black_pins[src_sq].no_block_king_directionbb = piece_move_list.no_block_king_directionbb;
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
        piece_move_list.attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= this->get_white_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        piece_move_list.movebb &= ~this->blackbb;

        // Pins and king attacks:
        {
            bitboard_t king_attackbb = bitboards::get_rook_movebb(src_sq, this->piecebbs[PieceType::WhiteKing]);
            if ((king_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
            {
                bitboard_t *directionbbs = bitboards::get_rook_magic(src_sq)->directionbbs;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[black_cardinal_direction_search_path[i]];
                    if ((directionbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                    {
                        piece_move_list.king_directionbb = directionbb & (king_attackbb | bitboards::get_sqbb(src_sq));
                        piece_move_list.no_block_king_directionbb = directionbb;
                        break;
                    }
                }

                bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::WhiteKing];
                bitboard_t pinbb = piece_move_list.king_directionbb & all_but_kingbb;

                // Only a true pin if just 1 piece is in attack path.
                if (bitboards::popcount(pinbb & ~bitboards::get_sqbb(src_sq)) == 1)
                {
                    this->black_pins[src_sq].pinbb = pinbb;
                    this->black_pins[src_sq].king_directionbb = piece_move_list.king_directionbb;
                    this->black_pins[src_sq].no_block_king_directionbb = piece_move_list.no_block_king_directionbb;
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
        piece_move_list.attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= this->get_white_pin_filterbb(bitboards::get_sqbb(src_sq));
        piece_move_list.movebb &= (this->checker_attackbb | this->checker_sqbb);
        piece_move_list.movebb &= (this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        piece_move_list.movebb &= ~this->blackbb;

        // Pins and king attacks:
        {
            bool king_in_diagonal_direction = false;
            // Diagonal:
            {
                bitboard_t king_attackbb = bitboards::get_bishop_movebb(src_sq, this->piecebbs[PieceType::WhiteKing]);
                if ((king_attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                {
                    bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                    for (int i = 0; i <= DirectionCnt; i++)
                    {
                        bitboard_t directionbb = directionbbs[black_diagonal_direction_search_path[i]];
                        if ((directionbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                        {
                            piece_move_list.king_directionbb = directionbb & (king_attackbb | bitboards::get_sqbb(src_sq));
                            piece_move_list.no_block_king_directionbb = directionbb;
                            king_in_diagonal_direction = true;
                            break;
                        }
                    }

                    bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::WhiteKing];
                    bitboard_t pinbb = piece_move_list.king_directionbb & all_but_kingbb;

                    // Only a true pin if just 1 piece is in attack path.
                    if (bitboards::popcount(pinbb & ~bitboards::get_sqbb(src_sq)) == 1)
                    {
                        this->black_pins[src_sq].pinbb = pinbb;
                        this->black_pins[src_sq].king_directionbb = piece_move_list.king_directionbb;
                        this->black_pins[src_sq].no_block_king_directionbb = piece_move_list.no_block_king_directionbb;
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
                    for (int i = 0; i <= DirectionCnt; i++)
                    {
                        bitboard_t directionbb = directionbbs[black_cardinal_direction_search_path[i]];
                        if ((directionbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
                        {
                            piece_move_list.king_directionbb = directionbb & (king_attackbb | bitboards::get_sqbb(src_sq));
                            piece_move_list.no_block_king_directionbb = directionbb;
                            break;
                        }
                    }

                    bitboard_t all_but_kingbb = this->allbb & ~this->piecebbs[PieceType::WhiteKing];
                    bitboard_t pinbb = piece_move_list.king_directionbb & all_but_kingbb;

                    // Only a true pin if just 1 piece is in attack path.
                    if (bitboards::popcount(pinbb & ~bitboards::get_sqbb(src_sq)) == 1)
                    {
                        this->black_pins[src_sq].pinbb = pinbb;
                        this->black_pins[src_sq].king_directionbb = piece_move_list.king_directionbb;
                        this->black_pins[src_sq].no_block_king_directionbb = piece_move_list.no_block_king_directionbb;
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
        piece_move_list.attackbb = bitboards::EmptyBB;
        this->black_king_attackbb = piece_move_list.movebb;

        piece_move_list.movebb &= ~this->white_attackbb;
        piece_move_list.movebb &= ~this->white_king_attackbb;

        // Checker bitboards are both initialized to full.
        if (this->checker_attackbb != bitboards::FullBB)
        {
            piece_move_list.movebb &= (~this->checker_attackbb | this->checker_sqbb);
        }
        if (this->discovered_checker_attackbb != bitboards::FullBB)
        {
            piece_move_list.movebb &= (~this->discovered_checker_attackbb | this->discovered_checker_sqbb);
        }

        piece_move_list.movebb &= ~this->blackbb;

        if ((this->checker_sqbb == bitboards::EmptyBB && this->discovered_checker_sqbb == bitboards::EmptyBB) &&
            this->castle_rights.black_left && bitboards::get_sqval(this->piecebbs[PieceType::BlackRook], 56) == 1)
        {
            bitboard_t left_betweenbb = bitboards::get_black_castle_left_betweenbb();
            if ((left_betweenbb & this->allbb) == bitboards::EmptyBB &&
                (left_betweenbb & this->white_attackbb) == bitboards::EmptyBB &&
                (left_betweenbb & this->white_king_attackbb) == bitboards::EmptyBB)
            {
                piece_move_list.movebb |= bitboards::get_sqbb(58);
            }
        }

        if ((this->checker_sqbb == bitboards::EmptyBB && this->discovered_checker_sqbb == bitboards::EmptyBB) &&
            this->castle_rights.black_right && bitboards::get_sqval(this->piecebbs[PieceType::BlackRook], 63) == 1)
        {
            bitboard_t right_betweenbb = bitboards::get_black_castle_right_betweenbb();
            if ((right_betweenbb & this->allbb) == bitboards::EmptyBB &&
                (right_betweenbb & this->white_attackbb) == bitboards::EmptyBB &&
                (right_betweenbb & this->white_king_attackbb) == bitboards::EmptyBB)
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
                    // Recent move could have invalidated pin. Need to make sure pin is still valid (pinner + pinned + king = 3).
                    if (bitboards::popcount(this->white_pins_trimmed[i]->king_directionbb & this->allbb) == 3)
                    {
                        pin_filterbb = this->white_pins_trimmed[i]->king_directionbb;
                        break;
                    }
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
                    // Recent move could have invalidated pin. Need to make sure pin is still valid (pinner + pinned + king = 3).
                    if (bitboards::popcount(this->black_pins_trimmed[i]->king_directionbb & this->allbb) == 3)
                    {
                        pin_filterbb = this->black_pins_trimmed[i]->king_directionbb;
                        break;
                    }
                }
            }
        }

        return pin_filterbb;
    }

    Pin *Position::get_white_discovered_check_pin(bitboard_t piecebb)
    {
        Pin *pin = nullptr;

        if ((piecebb & this->white_pinbb) != bitboards::EmptyBB)
        {
            for (int i = 0; i < this->white_pins_trimmed_cnt; i++)
            {
                if ((this->white_pins_trimmed[i]->king_directionbb & piecebb) != bitboards::EmptyBB)
                {
                    // Recent move will have invalidated pin. Need to make sure is real discovered check (pinner + king = 2).
                    if (bitboards::popcount(this->white_pins_trimmed[i]->king_directionbb & this->allbb) == 2)
                    {
                        pin = this->white_pins_trimmed[i];
                        break;
                    }
                }
            }
        }

        return pin;
    }

    Pin *Position::get_black_discovered_check_pin(bitboard_t piecebb)
    {
        Pin *pin = nullptr;

        if ((piecebb & this->black_pinbb) != bitboards::EmptyBB)
        {
            for (int i = 0; i < this->black_pins_trimmed_cnt; i++)
            {
                if ((this->black_pins_trimmed[i]->king_directionbb & piecebb) != bitboards::EmptyBB)
                {
                    // Recent move will have invalidated pin. Need to make sure is real discovered check (pinner + king = 2).
                    if (bitboards::popcount(this->black_pins_trimmed[i]->king_directionbb & this->allbb) == 2)
                    {
                        pin = this->black_pins_trimmed[i];
                        break;
                    }
                }
            }
        }

        return pin;
    }

    bitboard_t Position::get_white_direction_attackbb(square_t src_sq, bitboard_t piecebb)
    {
        bitboard_t attack_directionbb = bitboards::EmptyBB;

        {
            bool piece_in_diagonal_direction = false;

            // Diagonal:
            {
                bitboard_t attackbb = bitboards::get_bishop_movebb(src_sq, piecebb);

                bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[white_diagonal_direction_search_path[i]];
                    if ((directionbb & piecebb) != bitboards::EmptyBB)
                    {
                        attack_directionbb = directionbb & attackbb;
                        piece_in_diagonal_direction = true;
                        break;
                    }
                }
            }

            // Cardinal:
            if (!piece_in_diagonal_direction)
            {
                bitboard_t attackbb = bitboards::get_rook_movebb(src_sq, piecebb);

                bitboard_t *directionbbs = bitboards::get_rook_magic(src_sq)->directionbbs;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[white_cardinal_direction_search_path[i]];
                    if ((directionbb & piecebb) != bitboards::EmptyBB)
                    {
                        attack_directionbb = directionbb & attackbb;
                        break;
                    }
                }
            }
        }

        return attack_directionbb;
    }

    bitboard_t Position::get_black_direction_attackbb(square_t src_sq, bitboard_t piecebb)
    {
        bitboard_t attack_directionbb = bitboards::EmptyBB;

        {
            bool piece_in_diagonal_direction = false;

            // Diagonal:
            {
                bitboard_t attackbb = bitboards::get_bishop_movebb(src_sq, piecebb);

                bitboard_t *directionbbs = bitboards::get_bishop_magic(src_sq)->directionbbs;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[black_diagonal_direction_search_path[i]];
                    if ((directionbb & piecebb) != bitboards::EmptyBB)
                    {
                        attack_directionbb = directionbb & attackbb;
                        piece_in_diagonal_direction = true;
                        break;
                    }
                }
            }

            // Cardinal:
            if (!piece_in_diagonal_direction)
            {
                bitboard_t attackbb = bitboards::get_rook_movebb(src_sq, piecebb);

                bitboard_t *directionbbs = bitboards::get_rook_magic(src_sq)->directionbbs;
                for (int i = 0; i <= DirectionCnt; i++)
                {
                    bitboard_t directionbb = directionbbs[black_cardinal_direction_search_path[i]];
                    if ((directionbb & piecebb) != bitboards::EmptyBB)
                    {
                        attack_directionbb = directionbb & attackbb;
                        break;
                    }
                }
            }
        }

        return attack_directionbb;
    }

    MoveList Position::get_move_list()
    {
        MoveList move_list;
        move_list.move_cnt = 0;

        if (this->white_turn)
        {
            // Reset variables:
            this->white_attackbb = bitboards::EmptyBB;
            this->white_king_attackbb = bitboards::EmptyBB;
            memset(this->white_attackbbs, 0, sizeof(this->white_attackbbs));
            this->white_pinbb = bitboards::EmptyBB;
            memset(this->white_pins, 0, sizeof(this->white_pins));
            memset(this->white_pins_trimmed, 0, sizeof(this->white_pins_trimmed));
            this->white_pins_trimmed_cnt = 0;

            // WhitePawn:
            {
                PieceType piecetyp = PieceType::WhitePawn;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_pawn_moves(src_sq);

                    if ((bitboards::get_sqbb(src_sq) & bitboards::Row7BB) != bitboards::EmptyBB)
                    {
                        while (piece_move_list.movebb != bitboards::EmptyBB)
                        {
                            square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq, PieceType::WhiteKnight);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq, PieceType::WhiteBishop);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq, PieceType::WhiteRook);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq, PieceType::WhiteQueen);
                        }
                    }
                    else
                    {
                        while (piece_move_list.movebb != bitboards::EmptyBB)
                        {
                            square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                        }
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteKnight:
            {
                PieceType piecetyp = PieceType::WhiteKnight;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_knight_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteBishop:
            {
                PieceType piecetyp = PieceType::WhiteBishop;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_bishop_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteRook:
            {
                PieceType piecetyp = PieceType::WhiteRook;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_rook_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteQueen:
            {
                PieceType piecetyp = PieceType::WhiteQueen;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_queen_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }

                    this->white_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // WhiteKing:
            {
                PieceType piecetyp = PieceType::WhiteKing;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_white_king_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }
                }
            }
        }
        else
        {
            // Reset variables:
            this->black_attackbb = bitboards::EmptyBB;
            this->black_king_attackbb = bitboards::EmptyBB;
            memset(this->black_attackbbs, 0, sizeof(this->black_attackbbs));
            this->black_pinbb = bitboards::EmptyBB;
            memset(this->black_pins, 0, sizeof(this->black_pins));
            memset(this->black_pins_trimmed, 0, sizeof(this->black_pins_trimmed));
            this->black_pins_trimmed_cnt = 0;

            // BlackPawn:
            {
                PieceType piecetyp = PieceType::BlackPawn;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_pawn_moves(src_sq);

                    if ((bitboards::get_sqbb(src_sq) & bitboards::Row2BB) != bitboards::EmptyBB)
                    {
                        while (piece_move_list.movebb != bitboards::EmptyBB)
                        {
                            square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq, PieceType::BlackKnight);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq, PieceType::BlackBishop);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq, PieceType::BlackRook);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq, PieceType::BlackQueen);
                        }
                    }
                    else
                    {
                        while (piece_move_list.movebb != bitboards::EmptyBB)
                        {
                            square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                            move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                        }
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackKnight:
            {
                PieceType piecetyp = PieceType::BlackKnight;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_knight_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackBishop:
            {
                PieceType piecetyp = PieceType::BlackBishop;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_bishop_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackRook:
            {
                PieceType piecetyp = PieceType::BlackRook;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_rook_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackQueen:
            {
                PieceType piecetyp = PieceType::BlackQueen;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_queen_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }

                    this->black_attackbbs[src_sq] = piece_move_list.attackbb;
                }
            }

            // BlackKing:
            {
                PieceType piecetyp = PieceType::BlackKing;

                bitboard_t piecebb = this->piecebbs[piecetyp];
                while (piecebb != bitboards::EmptyBB)
                {
                    square_t src_sq = bitboards::pop_lsb(piecebb);

                    PieceMoveList piece_move_list = this->get_black_king_moves(src_sq);

                    while (piece_move_list.movebb != bitboards::EmptyBB)
                    {
                        square_t dst_sq = bitboards::pop_lsb(piece_move_list.movebb);
                        move_list.moves[move_list.move_cnt++] = Move(piecetyp, src_sq, dst_sq);
                    }
                }
            }
        }

        return move_list;
    }

    void Position::make_move(Move move)
    {
        // Reset variables:
        {
            this->checker_attackbb = bitboards::FullBB;
            this->checker_sqbb = bitboards::EmptyBB;
            this->discovered_checker_attackbb = bitboards::FullBB;
            this->discovered_checker_sqbb = bitboards::EmptyBB;
        }

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
        bool captured_au_passant = false;
        square_t capture_au_passant_sq;
        {
            bool au_passant_opportunity = false;

            if (src_piecetyp == PieceType::WhitePawn)
            {
                capture_au_passant_sq = move.dst_sq - 8;
                if (move.dst_sq == this->au_passant_sq)
                {
                    this->pieces[capture_au_passant_sq] = PieceType::None;
                    this->piecebbs[PieceType::BlackPawn] = bitboards::clear_sqval(this->piecebbs[PieceType::BlackPawn], capture_au_passant_sq);
                    this->blackbb = bitboards::clear_sqval(this->blackbb, capture_au_passant_sq);
                    captured_au_passant = true;
                }
                else if (move.dst_sq - move.src_sq > 9)
                {
                    this->au_passant_sq = capture_au_passant_sq;
                    au_passant_opportunity = true;
                }
            }
            else if (src_piecetyp == PieceType::BlackPawn)
            {
                capture_au_passant_sq = move.dst_sq + 8;
                if (move.dst_sq == this->au_passant_sq)
                {
                    this->pieces[capture_au_passant_sq] = PieceType::None;
                    this->piecebbs[PieceType::WhitePawn] = bitboards::clear_sqval(this->piecebbs[PieceType::WhitePawn], capture_au_passant_sq);
                    this->whitebb = bitboards::clear_sqval(this->whitebb, capture_au_passant_sq);
                    captured_au_passant = true;
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

        // Need to invalidate any moved pinners:
        {
            memset(&this->white_pins[move.src_sq], 0, sizeof(Pin));
            memset(&this->black_pins[move.src_sq], 0, sizeof(Pin));
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
            break;
        default:
            break;
        }

        // Attacks, pins, checks:
        if (this->white_turn)
        {
            // Update attacks and pins:
            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                // It is possible that recent move has invalidated an existing sliding attack(s):
                switch (this->pieces[sq])
                {
                case PieceType::WhiteBishop:
                {
                    if ((bitboards::get_sqbb(move.src_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB ||
                        (bitboards::get_sqbb(move.dst_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB ||
                        (captured_au_passant && (bitboards::get_sqbb(capture_au_passant_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB))
                    {
                        this->white_attackbbs[sq] = bitboards::get_bishop_movebb(sq, this->allbb);
                    }
                }
                break;
                case PieceType::WhiteRook:
                {
                    if ((bitboards::get_sqbb(move.src_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB ||
                        (bitboards::get_sqbb(move.dst_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB ||
                        (captured_au_passant && (bitboards::get_sqbb(capture_au_passant_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB))
                    {
                        this->white_attackbbs[sq] = bitboards::get_rook_movebb(sq, this->allbb);
                    }
                }
                break;
                case PieceType::WhiteQueen:
                {
                    if ((bitboards::get_sqbb(move.src_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB ||
                        (bitboards::get_sqbb(move.dst_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB ||
                        (captured_au_passant && (bitboards::get_sqbb(capture_au_passant_sq) & this->white_attackbbs[sq]) != bitboards::EmptyBB))
                    {
                        this->white_attackbbs[sq] = bitboards::get_bishop_movebb(sq, this->allbb);
                        this->white_attackbbs[sq] |= bitboards::get_rook_movebb(sq, this->allbb);
                    }
                }
                break;
                default:
                    break;
                }

                this->white_attackbb |= this->white_attackbbs[sq];

                this->white_pinbb |= this->white_pins[sq].pinbb;
                if (this->white_pins[sq].pinbb != bitboards::EmptyBB)
                {
                    this->white_pins_trimmed[this->white_pins_trimmed_cnt++] = &this->white_pins[sq];
                }
            }

            // See if moved piece is now checking opponent king:
            if ((moved_piece_move_list.attackbb & this->piecebbs[PieceType::BlackKing]) != bitboards::EmptyBB)
            {
                // If pawn or knight, there is no blocking opportunity. The only way to resolve the check is to take the piece or move the king.
                switch (this->pieces[move.dst_sq])
                {
                case PieceType::WhitePawn:
                case PieceType::WhiteKnight:
                    this->checker_sqbb = bitboards::get_sqbb(move.dst_sq);
                    this->checker_attackbb = bitboards::EmptyBB; // Set as empty since there is no blocking opportunity.
                    break;
                default:
                    this->checker_sqbb = bitboards::get_sqbb(move.dst_sq);
                    this->checker_attackbb = moved_piece_move_list.king_directionbb;

                    // Since we know piece checks king, we need to append attack bitboards with no block king direction.
                    /*
                    Example:    - bishop checks king (Northwest)
                                - bishop attack stops at king (blocker)
                                - king thinks moving further Northwest is valid since bishop attack does not go past king
                    */
                    this->white_attackbbs[move.src_sq] |= (moved_piece_move_list.no_block_king_directionbb & ~bitboards::get_sqbb(move.dst_sq));
                    this->white_attackbb |= this->white_attackbbs[move.src_sq];
                    break;
                }
            }

            // Discovered check:
            // TODO: au passant could lead to 2 discovered checks
            {
                Pin *white_discovered_check_pin = this->get_white_discovered_check_pin(bitboards::get_sqbb(move.src_sq));
                if (white_discovered_check_pin != nullptr &&
                    (bitboards::get_sqbb(move.dst_sq) & white_discovered_check_pin->king_directionbb) == bitboards::EmptyBB)
                {
                    this->discovered_checker_sqbb = bitboards::get_sqbb(white_discovered_check_pin->pinner_sq);
                    this->discovered_checker_attackbb = white_discovered_check_pin->king_directionbb;

                    // Since we know piece checks king, we need to append attack bitboards with no block king direction.
                    /*
                    Example:    - bishop checks king (Northwest)
                                - bishop attack stops at king (blocker)
                                - king thinks moving further Northwest is valid since bishop attack does not go past king
                    */
                    this->white_attackbbs[white_discovered_check_pin->pinner_sq] |= (white_discovered_check_pin->no_block_king_directionbb & ~bitboards::get_sqbb(white_discovered_check_pin->pinner_sq));
                    this->white_attackbb |= this->white_attackbbs[white_discovered_check_pin->pinner_sq];
                }
            }
        }
        else
        {
            // Update attacks and pins:
            for (square_t sq = 0; sq < SquareCnt; sq++)
            {
                // It is possible that recent move has invalidated an existing sliding attack:
                switch (this->pieces[sq])
                {
                case PieceType::BlackBishop:
                {
                    if ((bitboards::get_sqbb(move.src_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB ||
                        (bitboards::get_sqbb(move.dst_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB ||
                        (captured_au_passant && (bitboards::get_sqbb(capture_au_passant_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB))
                    {
                        this->black_attackbbs[sq] = bitboards::get_bishop_movebb(sq, this->allbb);
                    }
                }
                break;
                case PieceType::BlackRook:
                {
                    if ((bitboards::get_sqbb(move.src_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB ||
                        (bitboards::get_sqbb(move.dst_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB ||
                        (captured_au_passant && (bitboards::get_sqbb(capture_au_passant_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB))
                    {
                        this->black_attackbbs[sq] = bitboards::get_rook_movebb(sq, this->allbb);
                    }
                }
                break;
                case PieceType::BlackQueen:
                {
                    if ((bitboards::get_sqbb(move.src_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB ||
                        (bitboards::get_sqbb(move.dst_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB ||
                        (captured_au_passant && (bitboards::get_sqbb(capture_au_passant_sq) & this->black_attackbbs[sq]) != bitboards::EmptyBB))
                    {
                        this->black_attackbbs[sq] = bitboards::get_bishop_movebb(sq, this->allbb);
                        this->black_attackbbs[sq] |= bitboards::get_rook_movebb(sq, this->allbb);
                    }
                }
                break;
                default:
                    break;
                }

                this->black_attackbb |= this->black_attackbbs[sq];

                this->black_pinbb |= this->black_pins[sq].pinbb;
                if (this->black_pins[sq].pinbb != bitboards::EmptyBB)
                {
                    this->black_pins_trimmed[this->black_pins_trimmed_cnt++] = &this->black_pins[sq];
                }
            }

            // See if moved piece is now checking opponent king:
            if ((moved_piece_move_list.attackbb & this->piecebbs[PieceType::WhiteKing]) != bitboards::EmptyBB)
            {
                // If pawn or knight, there is no blocking opportunity. The only way to resolve the check is to take the piece or move the king.
                switch (this->pieces[move.dst_sq])
                {
                case PieceType::BlackPawn:
                case PieceType::BlackKnight:
                    this->checker_sqbb = bitboards::get_sqbb(move.dst_sq);
                    this->checker_attackbb = bitboards::EmptyBB; // Set as empty since there is no blocking opportunity.
                    break;
                default:
                    this->checker_sqbb = bitboards::get_sqbb(move.dst_sq);
                    this->checker_attackbb = moved_piece_move_list.king_directionbb;

                    // Since we know piece checks king, we need to append attack bitboards with no block king direction.
                    /*
                    Example:    - bishop checks king (Northwest)
                                - bishop attack stops at king (blocker)
                                - king thinks moving further Northwest is valid since bishop attack does not go past king
                    */
                    this->black_attackbbs[move.src_sq] |= (moved_piece_move_list.no_block_king_directionbb & ~bitboards::get_sqbb(move.dst_sq));
                    this->black_attackbb |= this->black_attackbbs[move.src_sq];
                    break;
                }
            }

            // Discovered check:
            // TODO: au passant could lead to 2 discovered checks
            {
                Pin *black_discovered_check_pin = this->get_black_discovered_check_pin(bitboards::get_sqbb(move.src_sq));
                if (black_discovered_check_pin != nullptr &&
                    (bitboards::get_sqbb(move.dst_sq) & black_discovered_check_pin->king_directionbb) == bitboards::EmptyBB)
                {
                    this->discovered_checker_sqbb = bitboards::get_sqbb(black_discovered_check_pin->pinner_sq);
                    this->discovered_checker_attackbb = black_discovered_check_pin->king_directionbb;

                    // Since we know piece checks king, we need to append attack bitboards with no block king direction.
                    /*
                    Example:    - bishop checks king (Northwest)
                                - bishop attack stops at king (blocker)
                                - king thinks moving further Northwest is valid since bishop attack does not go past king
                    */
                    this->black_attackbbs[black_discovered_check_pin->pinner_sq] |= (black_discovered_check_pin->no_block_king_directionbb & ~bitboards::get_sqbb(black_discovered_check_pin->pinner_sq));
                    this->black_attackbb |= this->black_attackbbs[black_discovered_check_pin->pinner_sq];
                }
            }
        }

        this->white_turn = !this->white_turn;
    }
}