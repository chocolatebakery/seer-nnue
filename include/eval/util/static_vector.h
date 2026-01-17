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
#include <cassert>
#include <cstddef>
#include <utility>

namespace eval::util {
template <typename T, usize Capacity>
class StaticVector {
 public:
  StaticVector() = default;
  ~StaticVector() = default;

  StaticVector(const StaticVector &other) { *this = other; }

  inline auto push(const T &elem) {
    assert(size_ < Capacity);
    data_[size_++] = elem;
  }

  inline auto push(T &&elem) {
    assert(size_ < Capacity);
    data_[size_++] = std::move(elem);
  }

  inline auto clear() { size_ = 0; }
  inline auto fill(const T &v) { data_.fill(v); }

  [[nodiscard]] inline auto size() const { return size_; }
  [[nodiscard]] inline auto empty() const { return size_ == 0; }

  [[nodiscard]] inline auto operator[](usize i) const -> const auto & {
    assert(i < size_);
    return data_[i];
  }

  [[nodiscard]] inline auto operator[](usize i) -> auto & {
    assert(i < size_);
    return data_[i];
  }

  [[nodiscard]] inline auto begin() { return data_.begin(); }
  [[nodiscard]] inline auto end() { return data_.begin() + static_cast<std::ptrdiff_t>(size_); }

  [[nodiscard]] inline auto begin() const { return data_.begin(); }
  [[nodiscard]] inline auto end() const { return data_.begin() + static_cast<std::ptrdiff_t>(size_); }

  inline auto resize(usize size) {
    assert(size <= Capacity);
    size_ = size;
  }

  inline auto operator=(const StaticVector &other) -> auto & {
    for (usize i = 0; i < other.size_; ++i) {
      data_[i] = other.data_[i];
    }
    size_ = other.size_;
    return *this;
  }

 private:
  std::array<T, Capacity> data_{};
  usize size_{0};
};
}  // namespace eval::util
