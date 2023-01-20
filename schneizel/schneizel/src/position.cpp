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

}