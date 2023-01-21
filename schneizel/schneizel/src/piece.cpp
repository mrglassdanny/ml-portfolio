#include "piece.h"

namespace schneizel
{
    char get_piecetyp_char(PieceType piecetyp)
    {
        switch (piecetyp)
        {
        case PieceType::WhitePawn:
            return 'P';
        case PieceType::WhiteKnight:
            return 'N';
        case PieceType::WhiteBishop:
            return 'B';
        case PieceType::WhiteRook:
            return 'R';
        case PieceType::WhiteQueen:
            return 'Q';
        case PieceType::WhiteKing:
            return 'K';
        case PieceType::BlackPawn:
            return 'p';
        case PieceType::BlackKnight:
            return 'n';
        case PieceType::BlackBishop:
            return 'b';
        case PieceType::BlackRook:
            return 'r';
        case PieceType::BlackQueen:
            return 'q';
        case PieceType::BlackKing:
            return 'k';
        default:
            return ' ';
        }
    }

    const char *get_piecetyp_str(PieceType piecetyp)
    {
        switch (piecetyp)
        {
        case PieceType::WhitePawn:
            return "WhitePawn";
        case PieceType::WhiteKnight:
            return "WhiteKnight";
        case PieceType::WhiteBishop:
            return "WhiteBishop";
        case PieceType::WhiteRook:
            return "WhiteRook";
        case PieceType::WhiteQueen:
            return "WhiteQueen";
        case PieceType::WhiteKing:
            return "WhiteKing";
        case PieceType::BlackPawn:
            return "BlackPawn";
        case PieceType::BlackKnight:
            return "BlackKnight";
        case PieceType::BlackBishop:
            return "BlackBishop";
        case PieceType::BlackRook:
            return "BlackRook";
        case PieceType::BlackQueen:
            return "BlackQueen";
        case PieceType::BlackKing:
            return "BlackKing";
        default:
            return "None";
        }
    }
}