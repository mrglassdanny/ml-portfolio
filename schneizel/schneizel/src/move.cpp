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
    }

    Move::Move(PieceType piecetyp, square_t src_sq, square_t dst_sq, PieceType promo_piecetyp)
    {
        this->piecetyp = piecetyp;
        this->src_sq = src_sq;
        this->dst_sq = dst_sq;
        this->promo_piecetyp = promo_piecetyp;
    }
}