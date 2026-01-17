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

#include <array>
#include <span>

namespace eval::util {
template <usize Alignment, typename T, usize Count>
class AlignedArray {
 public:
  [[nodiscard]] constexpr auto at(usize idx) -> auto & { return array_.at(idx); }
  [[nodiscard]] constexpr auto at(usize idx) const -> const auto & { return array_.at(idx); }

  [[nodiscard]] constexpr auto operator[](usize idx) -> auto & { return array_[idx]; }
  [[nodiscard]] constexpr auto operator[](usize idx) const -> const auto & { return array_[idx]; }

  [[nodiscard]] constexpr auto front() -> auto & { return array_.front(); }
  [[nodiscard]] constexpr auto front() const -> const auto & { return array_.front(); }

  [[nodiscard]] constexpr auto back() -> auto & { return array_.back(); }
  [[nodiscard]] constexpr auto back() const -> const auto & { return array_.back(); }

  [[nodiscard]] constexpr auto data() { return array_.data(); }
  [[nodiscard]] constexpr auto data() const { return array_.data(); }

  [[nodiscard]] constexpr auto begin() { return array_.begin(); }
  [[nodiscard]] constexpr auto begin() const { return array_.begin(); }
  [[nodiscard]] constexpr auto cbegin() const { return array_.cbegin(); }

  [[nodiscard]] constexpr auto end() { return array_.end(); }
  [[nodiscard]] constexpr auto end() const { return array_.end(); }
  [[nodiscard]] constexpr auto cend() const { return array_.cend(); }

  [[nodiscard]] constexpr auto rbegin() { return array_.rbegin(); }
  [[nodiscard]] constexpr auto rbegin() const { return array_.rbegin(); }
  [[nodiscard]] constexpr auto crbegin() const { return array_.crbegin(); }

  [[nodiscard]] constexpr auto rend() { return array_.rend(); }
  [[nodiscard]] constexpr auto rend() const { return array_.rend(); }
  [[nodiscard]] constexpr auto crend() const { return array_.crend(); }

  [[nodiscard]] constexpr auto empty() const { return array_.empty(); }
  [[nodiscard]] constexpr auto size() const { return array_.size(); }
  [[nodiscard]] constexpr auto max_size() const { return array_.max_size(); }

  constexpr auto fill(const T &value) { array_.fill(value); }
  constexpr auto swap(AlignedArray &other) { array_.swap(other.array_); }

  [[nodiscard]] constexpr auto array() -> auto & { return array_; }
  [[nodiscard]] constexpr auto array() const -> const auto & { return array_; }

  constexpr operator std::array<T, Count> &() { return array_; }
  constexpr operator const std::array<T, Count> &() const { return array_; }

  constexpr operator std::span<T, Count>() { return array_; }
  constexpr operator std::span<const T, Count>() const { return array_; }

 private:
  alignas(Alignment) std::array<T, Count> array_{};
};

template <usize Alignment, typename T, usize Count>
auto swap(AlignedArray<Alignment, T, Count> &a, AlignedArray<Alignment, T, Count> &b) {
  a.swap(b);
}
}  // namespace eval::util
