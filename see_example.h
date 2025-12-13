/*
 * Stormphrax Atomic SEE v2
 * - Fix quiet attacker bug
 * - Add king-shield weighting (cheap)
 * - Add gated 1-ply indirect king-kill scan (cheap)
 *
 * Designed for FICS Atomic rules:
 * - capture explodes capturer + adjacent non-pawns
 * - kings may touch
 * - move is illegal if own king explodes
 * - exploding enemy king wins immediately
 */

#pragma once

#include "types.h"
#include <array>

#include "core.h"
#include "position/position.h"
#include "attacks/attacks.h"

namespace stormphrax::see {

namespace values {
    constexpr Score Pawn   = 100;
    constexpr Score Knight = 450;
    constexpr Score Bishop = 450;
    constexpr Score Rook   = 650;
    constexpr Score Queen  = 1250;
    constexpr Score King   = 0;
}

constexpr auto Values = std::array{
    values::Pawn, values::Pawn,
    values::Knight, values::Knight,
    values::Bishop, values::Bishop,
    values::Rook, values::Rook,
    values::Queen, values::Queen,
    values::King, values::King,
    static_cast<Score>(0)
};

constexpr Score value(Piece p) {
    return Values[static_cast<i32>(p)];
}
constexpr Score value(PieceType pt) {
    return Values[static_cast<i32>(pt) * 2];
}

// ---------------------------
// Helpers
// ---------------------------

// Return king ring (adjacent squares) as Bitboard.
// If king missing (shouldn't happen), returns empty.
inline Bitboard kingRing(const BitboardSet& bbs, Color c) {
    Bitboard kbb = bbs.kings(c);
    if (kbb.empty())
        return Bitboard{};
    Square ks = static_cast<Square>(util::ctz(kbb));
    return attacks::getKingAttacks(ks);
}

// Check if opponent has an immediate capture on any occupied square
// adjacent to our king after the move, which would explode our king
// on that square. This is a very cheap 1-ply indirect kill scan.
// We only call it when our king ring was weakened.
inline bool immediateIndirectKill(const Position& pos,
                                  Color them,
                                  Bitboard occupiedAfter,
                                  Bitboard ourRing) {
    // only occupied squares in ring can be captured
    Bitboard ringTargets = ourRing & occupiedAfter;

    while (ringTargets) {
        Square r = static_cast<Square>(util::ctz(ringTargets));
        ringTargets &= ringTargets - 1;

        Bitboard atks = pos.attackersToPos(r, occupiedAfter, them);
        if (!atks.empty())
            return true;
    }
    return false;
}

// ---------------------------
// gain() for captures (MV/Fairy style)
// ---------------------------
inline Score gain(const PositionBoards& boards, Move move) {
    const auto& bbs = boards.bbs();

    Color us   = pieceColor(boards.pieceAt(move.src()));
    Color them = oppColor(us);

    Score score = 0;

    Bitboard fromTo =
        Bitboard::fromSquare(move.dst()) |
        Bitboard::fromSquare(move.src());

    if (move.type() == MoveType::EnPassant) {
        fromTo = Bitboard::fromSquare(move.src());
        score += value(colorPiece(PieceType::Pawn, them));
    }

    Bitboard boom =
        (attacks::getKingAttacks(move.dst()) & ~bbs.pawns()) |
        fromTo;

    Bitboard ourPieces   = bbs.occupancy(us);
    Bitboard theirPieces = bbs.occupancy(them);

    Bitboard boomUs   = boom & ourPieces;
    Bitboard boomThem = boom & theirPieces;

    if (boom & bbs.kings(us))
        return -ScoreMate;
    if (boom & bbs.kings(them))
        return ScoreMate;

    while (boomUs) {
        Square s = static_cast<Square>(util::ctz(boomUs));
        boomUs &= boomUs - 1;
        score -= value(boards.pieceAt(s));
    }
    while (boomThem) {
        Square s = static_cast<Square>(util::ctz(boomThem));
        boomThem &= boomThem - 1;
        score += value(boards.pieceAt(s));
    }

    return score;
}

// ---------------------------
// Atomic SEE scalar
// ---------------------------
inline Score gain_atomic(const Position& pos, Move move) {

    const auto& boards = pos.boards();
    const auto& bbs    = boards.bbs();

    Piece mover = boards.pieceAt(move.src());
    Color stm   = pieceColor(mover);
    Color them  = oppColor(stm);

    bool castle = move.type() == MoveType::Castling;

    Bitboard fromTo =
        Bitboard::fromSquare(move.dst()) |
        Bitboard::fromSquare(move.src());

    Piece captured = boards.pieceAt(move.dst());

    if (move.type() == MoveType::EnPassant) {
        fromTo   = Bitboard::fromSquare(move.src());
        captured = colorPiece(PieceType::Pawn, them);
    }

    Score result = 0;

    // ---------------- QUIET MOVES ----------------
    if (captured == Piece::None || castle) {

        Bitboard ourPieces   = bbs.occupancy(stm);
        Bitboard theirPieces = bbs.occupancy(them);

        Bitboard boom =
            (attacks::getKingAttacks(move.dst()) & ~bbs.pawns()) |
            (fromTo & bbs.occupancy());

        Bitboard boomUs   = boom & ourPieces;
        Bitboard boomThem = boom & theirPieces;

        Bitboard occupiedAfter = bbs.occupancy() ^ fromTo;

        // FIX: find least valuable attacker ONCE
        Bitboard attackers =
            pos.attackersToPos(move.dst(), occupiedAfter, them);

        Score minAttacker = ScoreMaxMate;

        while (attackers) {
            Square s = static_cast<Square>(util::ctz(attackers));
            attackers &= attackers - 1;

            if (pieceType(boards.pieceAt(s)) == PieceType::King)
                continue;

            Score v =
                (boom & Bitboard::fromSquare(s)) ? 0
                                                 : value(boards.pieceAt(s));

            minAttacker = std::min(minAttacker, v);
        }

        if (minAttacker != ScoreMaxMate)
            result += minAttacker;

        // King explosions decide immediately
        if (boom & bbs.kings(stm))
            return std::min(result - ScoreMate, 0);

        if (boom & bbs.kings(them))
            return std::min(result + ScoreMate, 0);

        // Atomic-aware shield weighting
        Bitboard ourRing   = kingRing(bbs, stm);
        Bitboard theirRing = kingRing(bbs, them);

        while (boomUs) {
            Square s = static_cast<Square>(util::ctz(boomUs));
            boomUs &= boomUs - 1;

            Piece p = boards.pieceAt(s);
            Score v = value(p);

            if (ourRing & Bitboard::fromSquare(s))
                v *= 4; // losing own king shield is very bad

            result -= v;
        }

        while (boomThem) {
            Square s = static_cast<Square>(util::ctz(boomThem));
            boomThem &= boomThem - 1;

            Piece p = boards.pieceAt(s);
            Score v = value(p);

            if (theirRing & Bitboard::fromSquare(s))
                v *= 3; // blowing enemy king shield is very good

            result += v;
        }

        // Gated indirect-kill scan:
        // only if our ring got weakened by this boom
        if (!ourRing.empty() && (boom & ourRing)) {
            if (immediateIndirectKill(pos, them, occupiedAfter, ourRing))
                result -= ScoreMate / 2; // huge negative but not "mate"
        }

        return std::min(result, 0);
    }

    // ---------------- CAPTURES ----------------
    if (captured != Piece::None && !castle) {
        result += gain(boards, move);

        // keep captures cheap; NNUE/search will handle deeper tactics
        return result - 1;
    }

    return std::min(result, 0);
}

// Public SEE gate (boolean)
inline bool see(const Position& pos, Move move, Score threshold = 0) {
    return gain_atomic(pos, move) >= threshold;
}

} // namespace stormphrax::see
