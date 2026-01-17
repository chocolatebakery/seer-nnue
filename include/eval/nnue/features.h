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

#include <algorithm>
#include <array>

#include <chess/square.h>
#include <chess/types.h>

namespace eval::nnue::features {
struct [[maybe_unused]] SingleBucket {
  static constexpr u32 BucketCount = 1;

  static constexpr auto getBucket([[maybe_unused]] chess::color c, [[maybe_unused]] chess::square kingSq) {
    return 0U;
  }

  static constexpr auto refreshRequired([[maybe_unused]] chess::color c,
      [[maybe_unused]] chess::square prevKingSq, [[maybe_unused]] chess::square kingSq) {
    return false;
  }
};

template <u32... BucketIndices>
struct [[maybe_unused]] KingBuckets {
  static_assert(sizeof...(BucketIndices) == 64);

 private:
  static constexpr auto Buckets = std::array{BucketIndices...};

 public:
  static constexpr auto BucketCount =
      static_cast<u32>(*std::max_element(Buckets.begin(), Buckets.end())) + 1U;

  static inline auto getBucket(const chess::color c, const chess::square kingSq) -> u32 {
    auto idx = square_index(kingSq);
    if (c == chess::color::black) {
      idx = flip_rank_index(idx);
    }
    return Buckets[idx];
  }

  static inline auto refreshRequired(const chess::color c, chess::square prevKingSq, chess::square kingSq) -> bool {
    auto prev = square_index(prevKingSq);
    auto next = square_index(kingSq);
    if (c == chess::color::black) {
      prev = flip_rank_index(prev);
      next = flip_rank_index(next);
    }
    return Buckets[prev] != Buckets[next];
  }
};
}  // namespace eval::nnue::features
