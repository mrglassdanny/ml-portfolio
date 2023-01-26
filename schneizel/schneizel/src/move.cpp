#include "move.h"

namespace schneizel
{
    Move::Move() {}

    Move::Move(PieceType piecetyp, square_t src_sq, square_t dst_sq)
    {
        this->piecetyp = piecetyp;
        this->src_sq = src_sq;
        this->dst_sq = dst_sq;
        this->promo_piecetyp = PieceType::None;
        this->gives_check = false;
    }

    Move::Move(PieceType piecetyp, square_t src_sq, square_t dst_sq, PieceType promo_piecetyp)
    {
        this->piecetyp = piecetyp;
        this->src_sq = src_sq;
        this->dst_sq = dst_sq;
        this->promo_piecetyp = promo_piecetyp;
        this->gives_check = false;
    }

    Move::Move(PieceType piecetyp, square_t src_sq, square_t dst_sq, bool gives_check)
    {
        this->piecetyp = piecetyp;
        this->src_sq = src_sq;
        this->dst_sq = dst_sq;
        this->promo_piecetyp = PieceType::None;
        this->gives_check = gives_check;
    }

    Move::Move(PieceType piecetyp, square_t src_sq, square_t dst_sq, PieceType promo_piecetyp, bool gives_check)
    {
        this->piecetyp = piecetyp;
        this->src_sq = src_sq;
        this->dst_sq = dst_sq;
        this->promo_piecetyp = promo_piecetyp;
        this->gives_check = gives_check;
    }
}