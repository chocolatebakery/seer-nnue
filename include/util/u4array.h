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

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace util {

class IndexedU4 {
 public:
  constexpr inline operator std::uint8_t() const { return high_ ? (value_ >> 4) : (value_ & 0xF); }

  constexpr inline IndexedU4& operator=(std::uint8_t v) {
    assert(v <= 0xF);
    if (high_) {
      value_ = static_cast<std::uint8_t>((value_ & 0x0F) | (v << 4));
    } else {
      value_ = static_cast<std::uint8_t>((value_ & 0xF0) | (v & 0x0F));
    }
    return *this;
  }

 private:
  constexpr IndexedU4(std::uint8_t& value, bool high) : value_{value}, high_{high} {}

  std::uint8_t& value_;
  bool high_;

  template <std::size_t Size>
  friend class U4Array;
};

template <std::size_t Size>
class U4Array {
  static_assert(Size % 2 == 0, "U4Array size must be even");

 public:
  U4Array() = default;
  ~U4Array() = default;

  constexpr std::uint8_t operator[](std::size_t i) const {
    assert(i < Size);
    return static_cast<std::uint8_t>(data_[i / 2] >> ((i % 2) * 4));
  }

  constexpr IndexedU4 operator[](std::size_t i) {
    assert(i < Size);
    return IndexedU4{data_[i / 2], (i % 2) == 1};
  }

 private:
  std::array<std::uint8_t, Size / 2> data_{};
};

}  // namespace util
