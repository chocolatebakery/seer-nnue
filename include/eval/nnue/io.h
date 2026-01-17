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
#include <istream>
#include <ostream>
#include <span>
#include <type_traits>
#include <variant>

namespace eval::nnue {
class IParamStream {
 public:
  virtual ~IParamStream() = default;

  template <typename T>
  inline auto read(std::span<T> dst) -> bool {
    if constexpr (std::is_same_v<T, i16>) {
      return readI16s(dst);
    } else {
      static_assert(std::is_same_v<T, i16>, "unsupported parameter type");
      return false;
    }
  }

  template <typename T, usize Size>
  inline auto read(std::array<T, Size> &dst) {
    return read(std::span<T, std::dynamic_extent>{dst});
  }

  template <typename T>
  inline auto write(std::span<const T> src) -> bool {
    if constexpr (std::is_same_v<T, i16>) {
      return writeI16s(src);
    } else {
      static_assert(std::is_same_v<T, i16>, "unsupported parameter type");
      return false;
    }
  }

  template <typename T, usize Size>
  inline auto write(const std::array<T, Size> &src) {
    return write(std::span<const T, std::dynamic_extent>{src});
  }

 protected:
  virtual auto readI16s(std::span<i16> dst) -> bool = 0;
  virtual auto writeI16s(std::span<const i16> src) -> bool = 0;
};

template <usize BlockSize>
class PaddedParamStream final : public IParamStream {
 public:
  explicit PaddedParamStream(std::istream &in) : stream_{&in} {}
  explicit PaddedParamStream(std::ostream &out) : stream_{&out} {}

  ~PaddedParamStream() final = default;

 protected:
  inline auto readI16s(std::span<i16> dst) -> bool final { return read(dst.data(), dst.size_bytes()); }
  inline auto writeI16s(std::span<const i16> src) -> bool final { return write(src.data(), src.size_bytes()); }

 private:
  std::variant<std::istream *, std::ostream *> stream_;

  inline auto read(void *dst, usize n) -> bool {
    if (!std::holds_alternative<std::istream *>(stream_)) {
      assert(false);
      return false;
    }

    auto &stream = *std::get<std::istream *>(stream_);
    const auto padding = calcPadding(n);

    stream.read(static_cast<char *>(dst), static_cast<std::streamsize>(n));
    stream.ignore(static_cast<std::streamsize>(padding));
    return !stream.fail();
  }

  inline auto write(const void *src, usize n) -> bool {
    if (!std::holds_alternative<std::ostream *>(stream_)) {
      assert(false);
      return false;
    }

    static constexpr std::array<std::byte, BlockSize> Empty{};
    auto &stream = *std::get<std::ostream *>(stream_);

    const auto padding = calcPadding(n);
    stream.write(static_cast<const char *>(src), static_cast<std::streamsize>(n));
    stream.write(reinterpret_cast<const char *>(Empty.data()), padding);
    return !stream.fail();
  }

  [[nodiscard]] static constexpr auto calcPadding(usize v) -> usize {
    return v - ((v + BlockSize - 1) / BlockSize) * BlockSize;
  }
};
}  // namespace eval::nnue
