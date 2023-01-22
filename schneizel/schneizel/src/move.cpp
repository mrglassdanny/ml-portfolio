#include "move.h"

namespace schneizel
{
    Move::Move() {}

    Move::Move(PieceType piecetyp, byte_t src_sqnum, byte_t dst_sqnum)
    {
        this->piecetyp = piecetyp;
        this->src_sqnum = src_sqnum;
        this->dst_sqnum = dst_sqnum;
        this->promo_piecetyp = PieceType::None;
    }

    Move::Move(PieceType piecetyp, byte_t src_sqnum, byte_t dst_sqnum, PieceType promo_piecetyp)
    {
        this->piecetyp = piecetyp;
        this->src_sqnum = src_sqnum;
        this->dst_sqnum = dst_sqnum;
        this->promo_piecetyp = promo_piecetyp;
    }
}