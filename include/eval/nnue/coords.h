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
#include <chess/square.h>
#include <chess/types.h>

namespace eval::nnue {
inline constexpr auto color_index(const chess::color c) -> int {
  return c == chess::color::white ? 0 : 1;
}

inline constexpr auto piece_index(const chess::piece_type pt) -> int {
  return static_cast<int>(pt);
}

inline constexpr auto flip_file_index(const u32 idx) -> u32 {
  return idx ^ 7U;
}

inline constexpr auto flip_rank_index(const u32 idx) -> u32 {
  return idx ^ 0x38U;
}

inline auto square_index(const chess::square &sq) -> u32 {
  return static_cast<u32>(sq.index()) ^ 7U;
}

inline auto feature_square_index(const chess::square &sq, const chess::color perspective) -> u32 {
  auto idx = square_index(sq);
  if (perspective == chess::color::black) {
    idx = flip_rank_index(idx);
  }
  return idx;
}
}  // namespace eval::nnue
