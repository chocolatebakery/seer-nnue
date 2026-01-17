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

#include <cassert>
#include <cstddef>
#include <istream>
#include <limits>
#include <span>

namespace eval::util {
class MemoryBuffer : public std::streambuf {
 public:
  explicit MemoryBuffer(std::span<const std::byte> data)
      : begin_{reinterpret_cast<char *>(const_cast<std::byte *>(data.data()))},
        end_{begin_ + data.size_bytes()} {
    setg(begin_, begin_, end_);
  }

  auto seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) -> pos_type override {
    (void)which;
    assert(off <= std::numeric_limits<i32>::max());

    switch (dir) {
      case std::ios_base::cur:
        gbump(static_cast<i32>(off));
        break;
      case std::ios_base::end:
        setg(begin_, end_ + off, end_);
        break;
      case std::ios_base::beg:
        setg(begin_, begin_ + off, end_);
        break;
      default:
        break;
    }

    return gptr() - eback();
  }

  auto seekpos(std::streampos pos, std::ios_base::openmode mode) -> pos_type override {
    return seekoff(pos, std::ios_base::beg, mode);
  }

 private:
  char *begin_;
  char *end_;
};

class MemoryIstream : public std::istream {
 public:
  explicit MemoryIstream(std::span<const std::byte> data) : std::istream(nullptr), buf_{data} {
    rdbuf(&buf_);
  }

 private:
  MemoryBuffer buf_;
};
}  // namespace eval::util
