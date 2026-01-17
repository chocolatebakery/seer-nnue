/*
 * Stormphrax, a UCI chess engine
 * Copyright (C) 2024 Ciekce
 *
 * Stormphrax is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Stormphrax is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Stormphrax. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include "../../eval/types.h"
#include "coords.h"

#include <chess/board.h>
#include <chess/piece_configuration.h>
#include <chess/square.h>
#include <chess/types.h>

#include <array>

namespace eval::nnue {
class BitboardSet {
 public:
  BitboardSet() = default;

  [[nodiscard]] inline auto forColor(const chess::color c) const { return colors_[color_index(c)]; }
  [[nodiscard]] inline auto forPiece(const chess::piece_type pt) const { return pieces_[piece_index(pt)]; }
  [[nodiscard]] inline auto forPiece(const chess::piece_type pt, const chess::color c) const {
    return pieces_[piece_index(pt)] & colors_[color_index(c)];
  }

  [[nodiscard]] inline auto blackOccupancy() const { return forColor(chess::color::black); }
  [[nodiscard]] inline auto whiteOccupancy() const { return forColor(chess::color::white); }
  [[nodiscard]] inline auto occupancy() const { return colors_[0] | colors_[1]; }

  [[nodiscard]] inline auto pawns() const { return forPiece(chess::piece_type::pawn); }
  [[nodiscard]] inline auto knights() const { return forPiece(chess::piece_type::knight); }
  [[nodiscard]] inline auto bishops() const { return forPiece(chess::piece_type::bishop); }
  [[nodiscard]] inline auto rooks() const { return forPiece(chess::piece_type::rook); }
  [[nodiscard]] inline auto queens() const { return forPiece(chess::piece_type::queen); }
  [[nodiscard]] inline auto kings() const { return forPiece(chess::piece_type::king); }

  [[nodiscard]] inline auto blackPawns() const { return pawns() & blackOccupancy(); }
  [[nodiscard]] inline auto whitePawns() const { return pawns() & whiteOccupancy(); }

  [[nodiscard]] inline auto blackKnights() const { return knights() & blackOccupancy(); }
  [[nodiscard]] inline auto whiteKnights() const { return knights() & whiteOccupancy(); }

  [[nodiscard]] inline auto blackBishops() const { return bishops() & blackOccupancy(); }
  [[nodiscard]] inline auto whiteBishops() const { return bishops() & whiteOccupancy(); }

  [[nodiscard]] inline auto blackRooks() const { return rooks() & blackOccupancy(); }
  [[nodiscard]] inline auto whiteRooks() const { return rooks() & whiteOccupancy(); }

  [[nodiscard]] inline auto blackQueens() const { return queens() & blackOccupancy(); }
  [[nodiscard]] inline auto whiteQueens() const { return queens() & whiteOccupancy(); }

  [[nodiscard]] inline auto blackKings() const { return kings() & blackOccupancy(); }
  [[nodiscard]] inline auto whiteKings() const { return kings() & whiteOccupancy(); }

  [[nodiscard]] inline auto minors() const { return knights() | bishops(); }
  [[nodiscard]] inline auto majors() const { return rooks() | queens(); }
  [[nodiscard]] inline auto nonPk() const { return occupancy() ^ pawns() ^ kings(); }

  [[nodiscard]] inline auto blackMinors() const { return minors() & blackOccupancy(); }
  [[nodiscard]] inline auto whiteMinors() const { return minors() & whiteOccupancy(); }

  [[nodiscard]] inline auto blackMajors() const { return majors() & blackOccupancy(); }
  [[nodiscard]] inline auto whiteMajors() const { return majors() & whiteOccupancy(); }

  [[nodiscard]] inline auto blackNonPk() const { return blackOccupancy() ^ ((pawns() | kings()) & blackOccupancy()); }
  [[nodiscard]] inline auto whiteNonPk() const { return whiteOccupancy() ^ ((pawns() | kings()) & whiteOccupancy()); }

  [[nodiscard]] inline auto pawns(const chess::color c) const { return forPiece(chess::piece_type::pawn, c); }
  [[nodiscard]] inline auto knights(const chess::color c) const { return forPiece(chess::piece_type::knight, c); }
  [[nodiscard]] inline auto bishops(const chess::color c) const { return forPiece(chess::piece_type::bishop, c); }
  [[nodiscard]] inline auto rooks(const chess::color c) const { return forPiece(chess::piece_type::rook, c); }
  [[nodiscard]] inline auto queens(const chess::color c) const { return forPiece(chess::piece_type::queen, c); }
  [[nodiscard]] inline auto kings(const chess::color c) const { return forPiece(chess::piece_type::king, c); }

  [[nodiscard]] inline auto minors(const chess::color c) const { return c == chess::color::black ? blackMinors() : whiteMinors(); }
  [[nodiscard]] inline auto majors(const chess::color c) const { return c == chess::color::black ? blackMajors() : whiteMajors(); }
  [[nodiscard]] inline auto nonPk(const chess::color c) const { return c == chess::color::black ? blackNonPk() : whiteNonPk(); }

  [[nodiscard]] inline auto operator==(const BitboardSet &other) const -> bool = default;

  [[nodiscard]] static inline auto from_board(const chess::board &bd) -> BitboardSet {
    BitboardSet bbs{};
    bbs.colors_[color_index(chess::color::white)] = bd.man_.white.all();
    bbs.colors_[color_index(chess::color::black)] = bd.man_.black.all();

    chess::over_types([&](const chess::piece_type pt) {
      bbs.pieces_[piece_index(pt)] = bd.man_.white.get_plane(pt) | bd.man_.black.get_plane(pt);
    });

    return bbs;
  }

 private:
  std::array<chess::square_set, 2> colors_{};
  std::array<chess::square_set, 6> pieces_{};
};

namespace boards {
constexpr chess::square_set DarkSquares{0x55AA55AA55AA55AAULL};
constexpr chess::square_set LightSquares{0xAA55AA55AA55AA55ULL};
}  // namespace boards
}  // namespace eval::nnue
