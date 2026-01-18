/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <chess/board.h>
#include <chess/castle_info.h>

#include <algorithm>
#include <array>
#include <cstdint>

namespace train {

enum class Outcome : std::uint8_t { WhiteLoss = 0, Draw, WhiteWin };

namespace marlinformat {

// https://github.com/jnlt3/marlinflow/blob/main/marlinformat/src/lib.rs
struct __attribute__((packed)) PackedBoard {
  std::uint64_t occupancy;
  std::uint8_t pieces[16];
  std::uint8_t stmEpSquare;
  std::uint8_t halfmoveClock;
  std::uint16_t fullmoveNumber;
  std::int16_t eval;
  Outcome wdl;
  std::uint8_t extra;

  [[nodiscard]] static PackedBoard pack(const chess::board& bd, std::int16_t score) {
    constexpr std::uint8_t UnmovedRook = 6;
    constexpr std::uint8_t NoEpSquare = 64;

    PackedBoard out{};

    const bool white_oo = bd.lat_.white.oo();
    const bool white_ooo = bd.lat_.white.ooo();
    const bool black_oo = bd.lat_.black.oo();
    const bool black_ooo = bd.lat_.black.ooo();

    const auto white_oo_sq = chess::castle_info<chess::color::white>.oo_rook;
    const auto white_ooo_sq = chess::castle_info<chess::color::white>.ooo_rook;
    const auto black_oo_sq = chess::castle_info<chess::color::black>.oo_rook;
    const auto black_ooo_sq = chess::castle_info<chess::color::black>.ooo_rook;

    std::array<std::uint8_t, 64> piece_map{};
    std::uint64_t occupancy = 0;

    const auto add_piece = [&](const chess::color color, const chess::piece_type pt, const chess::square& sq) {
      const int std_file = 7 - sq.file();
      const int std_rank = sq.rank();
      const int idx = std_rank * 8 + std_file;

      const bool is_unmoved_rook =
          (pt == chess::piece_type::rook) &&
          ((color == chess::color::white &&
            ((white_oo && sq == white_oo_sq) || (white_ooo && sq == white_ooo_sq))) ||
           (color == chess::color::black &&
            ((black_oo && sq == black_oo_sq) || (black_ooo && sq == black_ooo_sq))));

      const std::uint8_t base_id = is_unmoved_rook ? UnmovedRook : static_cast<std::uint8_t>(pt);
      const std::uint8_t color_id = (color == chess::color::black) ? static_cast<std::uint8_t>(1u << 3) : 0;
      piece_map[static_cast<std::size_t>(idx)] = static_cast<std::uint8_t>(base_id | color_id);
      occupancy |= (static_cast<std::uint64_t>(1) << static_cast<std::uint64_t>(idx));
    };

    chess::over_types([&](const chess::piece_type pt) {
      for (const auto sq : bd.man_.white.get_plane(pt)) { add_piece(chess::color::white, pt, sq); }
      for (const auto sq : bd.man_.black.get_plane(pt)) { add_piece(chess::color::black, pt, sq); }
    });

    out.occupancy = occupancy;

    auto set_piece = [&](const std::size_t idx, const std::uint8_t value) {
      auto& cell = out.pieces[idx / 2];
      const std::uint8_t nibble = static_cast<std::uint8_t>(value & 0x0F);
      if ((idx & 1U) == 0) {
        cell = static_cast<std::uint8_t>((cell & 0xF0) | nibble);
      } else {
        cell = static_cast<std::uint8_t>((cell & 0x0F) | (nibble << 4));
      }
    };

    std::uint64_t scan = occupancy;
    std::size_t i = 0;
    while (scan) {
      const int idx = static_cast<int>(chess::count_trailing_zeros(scan));
      set_piece(i++, piece_map[static_cast<std::size_t>(idx)]);
      scan &= (scan - 1);
    }

    const bool stm_white = bd.turn();
    const std::uint8_t stm = stm_white ? 0 : static_cast<std::uint8_t>(1u << 7);

    std::uint8_t ep_value = NoEpSquare;
    const auto ep_mask = bd.lat_.them(bd.turn()).ep_mask();
    if (ep_mask.any()) {
      const auto ep_sq = ep_mask.item();
      const int std_file = 7 - ep_sq.file();
      const int rank = stm_white ? 5 : 2;
      ep_value = static_cast<std::uint8_t>(rank * 8 + std_file);
    }

    out.stmEpSquare = static_cast<std::uint8_t>(stm | ep_value);
    out.halfmoveClock = static_cast<std::uint8_t>(std::min<std::size_t>(bd.lat_.half_clock, 255));
    out.fullmoveNumber = static_cast<std::uint16_t>(std::min<std::size_t>(1 + (bd.lat_.ply_count / 2), 65535));
    out.eval = score;

    return out;
  }
};

}  // namespace marlinformat

}  // namespace train
