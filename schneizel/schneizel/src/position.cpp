#include "position.h"

namespace schneizel
{
    void Position::init()
    {
        white_turn = true;

        // White pieces:
        {
            int piece_cnt = 0;

            for (byte_t sqnum = 8; sqnum < 16; sqnum++)
                this->white_pieces[piece_cnt++] = Piece{PieceType::Pawn, sqnum};

            this->white_pieces[piece_cnt++] = Piece{PieceType::Knight, 1};
            this->white_pieces[piece_cnt++] = Piece{PieceType::Knight, 6};

            this->white_pieces[piece_cnt++] = Piece{PieceType::Bishop, 2};
            this->white_pieces[piece_cnt++] = Piece{PieceType::Bishop, 5};

            this->white_pieces[piece_cnt++] = Piece{PieceType::Rook, 0};
            this->white_pieces[piece_cnt++] = Piece{PieceType::Rook, 7};

            this->white_pieces[piece_cnt++] = Piece{PieceType::Queen, 3};
            this->white_pieces[piece_cnt++] = Piece{PieceType::King, 4};
        }

        // Black pieces:
        {
            int piece_cnt = 0;

            for (byte_t sqnum = 48; sqnum < 56; sqnum++)
                this->black_pieces[piece_cnt++] = (Piece{PieceType::Pawn, sqnum});

            this->black_pieces[piece_cnt++] = Piece{PieceType::Knight, 57};
            this->black_pieces[piece_cnt++] = Piece{PieceType::Knight, 62};

            this->black_pieces[piece_cnt++] = Piece{PieceType::Bishop, 58};
            this->black_pieces[piece_cnt++] = Piece{PieceType::Bishop, 61};

            this->black_pieces[piece_cnt++] = Piece{PieceType::Rook, 56};
            this->black_pieces[piece_cnt++] = Piece{PieceType::Rook, 63};

            this->black_pieces[piece_cnt++] = Piece{PieceType::Queen, 59};
            this->black_pieces[piece_cnt++] = Piece{PieceType::King, 60};
        }

        // Bitboards:
        memset(this->white_bbs, 0, sizeof(this->white_bbs));
        memset(this->black_bbs, 0, sizeof(this->black_bbs));

        for (int i = 0; i < PieceMaxCnt; i++)
        {
            Piece *white_piece = &this->white_pieces[i];
            if (white_piece->typ != PieceType::None)
            {
                this->white_bbs[white_piece->typ] = bitboards::set_sqval(this->white_bbs[white_piece->typ], white_piece->sqnum);
            }

            Piece *black_piece = &this->black_pieces[i];
            if (black_piece->typ != PieceType::None)
            {
                this->black_bbs[black_piece->typ] = bitboards::set_sqval(this->black_bbs[black_piece->typ], black_piece->sqnum);
            }
        }
    }

    bitboard_t Position::get_whitebb()
    {
        bitboard_t bb = Empty;

        for (int i = 0; i < PieceTypeCnt; i++)
        {
            bb |= this->white_bbs[i];
        }

        return bb;
    }

    bitboard_t Position::get_blackbb()
    {
        bitboard_t bb = Empty;

        for (int i = 0; i < PieceTypeCnt; i++)
        {
            bb |= this->black_bbs[i];
        }

        return bb;
    }

    bitboard_t Position::get_allbb()
    {
        return this->get_whitebb() | this->get_blackbb();
    }

    Move Position::get_moves()
    {
        return Move{PieceType::None, 0, 0};
    }

    void Position::make_move(Move move)
    {
        if (this->white_turn)
        {
            for (int i = 0; i < PieceMaxCnt; i++)
            {
                if (move.piece_typ == this->white_pieces[i].typ)
                {
                    if (move.src_sqnum == this->white_pieces[i].sqnum)
                    {
                        this->white_pieces[i].sqnum = move.dst_sqnum;
                        this->white_bbs[this->white_pieces[i].typ] = set_sqval(this->white_bbs[this->white_pieces[i].typ], move.dst_sqnum);
                        this->white_bbs[this->white_pieces[i].typ] = clear_sqval(this->white_bbs[this->white_pieces[i].typ], move.src_sqnum);
                    }
                }

                if (move.dst_sqnum == this->black_pieces[i].sqnum)
                {
                    this->black_bbs[this->black_pieces[i].typ] = clear_sqval(this->black_bbs[this->black_pieces[i].typ], move.dst_sqnum);
                    this->black_pieces[i].typ = PieceType::None;
                }
            }
        }
        else
        {
            for (int i = 0; i < PieceMaxCnt; i++)
            {
                if (move.piece_typ == this->black_pieces[i].typ)
                {
                    if (move.src_sqnum == this->black_pieces[i].sqnum)
                    {
                        this->black_pieces[i].sqnum = move.dst_sqnum;
                        this->black_bbs[this->black_pieces[i].typ] = set_sqval(this->black_bbs[this->black_pieces[i].typ], move.dst_sqnum);
                        this->black_bbs[this->black_pieces[i].typ] = clear_sqval(this->black_bbs[this->black_pieces[i].typ], move.src_sqnum);
                    }
                }

                if (move.dst_sqnum == this->white_pieces[i].sqnum)
                {
                    this->white_bbs[this->white_pieces[i].typ] = clear_sqval(this->white_bbs[this->white_pieces[i].typ], move.dst_sqnum);
                    this->white_pieces[i].typ = PieceType::None;
                }
            }
        }

        this->white_turn = !this->white_turn;
    }
}