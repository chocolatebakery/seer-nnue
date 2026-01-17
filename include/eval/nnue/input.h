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
#include "../util/simd.h"
#include "bitboard_set.h"
#include "coords.h"
#include "features.h"
#include "io.h"

#include <array>
#include <algorithm>
#include <span>

namespace eval::nnue {
template <typename Ft>
class Accumulator {
 public:
  [[nodiscard]] inline auto black() const -> std::span<const typename Ft::OutputType, Ft::OutputCount> {
    return outputs_[color_index(chess::color::black)];
  }

  [[nodiscard]] inline auto white() const -> std::span<const typename Ft::OutputType, Ft::OutputCount> {
    return outputs_[color_index(chess::color::white)];
  }

  [[nodiscard]] inline auto forColor(chess::color c) const -> std::span<const typename Ft::OutputType, Ft::OutputCount> {
    return outputs_[color_index(c)];
  }

  [[nodiscard]] inline auto black() -> std::span<typename Ft::OutputType, Ft::OutputCount> {
    return outputs_[color_index(chess::color::black)];
  }

  [[nodiscard]] inline auto white() -> std::span<typename Ft::OutputType, Ft::OutputCount> {
    return outputs_[color_index(chess::color::white)];
  }

  [[nodiscard]] inline auto forColor(chess::color c) -> std::span<typename Ft::OutputType, Ft::OutputCount> {
    return outputs_[color_index(c)];
  }

  inline void initBoth(const Ft &featureTransformer) {
    std::copy(featureTransformer.biases.begin(), featureTransformer.biases.end(), outputs_[0].begin());
    std::copy(featureTransformer.biases.begin(), featureTransformer.biases.end(), outputs_[1].begin());
  }

  inline auto subAddFrom(Accumulator &src, const Ft &featureTransformer, chess::color c, u32 sub, u32 add) {
    subAdd(src.forColor(c), forColor(c), featureTransformer.weights, sub * OutputCount, add * OutputCount);
  }

  inline auto subFrom(Accumulator &src, const Ft &featureTransformer, chess::color c, std::span<const u32> subs) {
    subSub(src.forColor(c), forColor(c), featureTransformer.weights, subs, OutputCount);
  }

  inline auto subSubAddFrom(Accumulator &src, const Ft &featureTransformer, chess::color c, u32 sub0, u32 sub1, u32 add) {
    subSubAdd(src.forColor(c), forColor(c), featureTransformer.weights, sub0 * OutputCount, sub1 * OutputCount, add * OutputCount);
  }

  inline auto subSubAddAddFrom(
      Accumulator &src, const Ft &featureTransformer, chess::color c, u32 sub0, u32 sub1, u32 add0, u32 add1) {
    subSubAddAdd(src.forColor(c), forColor(c), featureTransformer.weights, sub0 * OutputCount, sub1 * OutputCount,
        add0 * OutputCount, add1 * OutputCount);
  }

  inline auto activateFeature(const Ft &featureTransformer, chess::color c, u32 feature) {
    add(forColor(c), featureTransformer.weights, feature * OutputCount);
  }

  inline auto deactivateFeature(const Ft &featureTransformer, chess::color c, u32 feature) {
    sub(forColor(c), featureTransformer.weights, feature * OutputCount);
  }

  inline auto copyFrom(chess::color c, const Accumulator &other) {
    const auto idx = color_index(c);
    std::copy(other.outputs_[idx].begin(), other.outputs_[idx].end(), outputs_[idx].begin());
  }

 private:
  using Type = typename Ft::OutputType;

  static constexpr auto InputCount = Ft::InputCount;
  static constexpr auto WeightCount = Ft::WeightCount;
  static constexpr auto OutputCount = Ft::OutputCount;

  EVAL_SIMD_ALIGNAS std::array<std::array<Type, OutputCount>, 2> outputs_{};

  static inline auto subAdd(std::span<Type, OutputCount> src, std::span<Type, OutputCount> dst,
      std::span<const Type, WeightCount> delta, u32 subOffset, u32 addOffset) -> void {
    for (u32 i = 0; i < OutputCount; ++i) {
      dst[i] = src[i] + delta[addOffset + i] - delta[subOffset + i];
    }
  }

  static inline auto subSub(std::span<Type, OutputCount> src, std::span<Type, OutputCount> dst,
      std::span<const Type, WeightCount> delta, std::span<const u32> subs, u32 stride) -> void {
    for (u32 i = 0; i < OutputCount; ++i) {
      dst[i] = src[i];
      for (usize j = 0; j < subs.size(); ++j) {
        dst[i] -= delta[subs[j] * stride + i];
      }
    }
  }

  static inline auto subSubAdd(std::span<Type, OutputCount> src, std::span<Type, OutputCount> dst,
      std::span<const Type, WeightCount> delta, u32 subOffset0, u32 subOffset1, u32 addOffset) -> void {
    for (u32 i = 0; i < OutputCount; ++i) {
      dst[i] = src[i] + delta[addOffset + i] - delta[subOffset0 + i] - delta[subOffset1 + i];
    }
  }

  static inline auto subSubAddAdd(std::span<Type, OutputCount> src, std::span<Type, OutputCount> dst,
      std::span<const Type, WeightCount> delta, u32 subOffset0, u32 subOffset1, u32 addOffset0, u32 addOffset1) -> void {
    for (u32 i = 0; i < OutputCount; ++i) {
      dst[i] = src[i] + delta[addOffset0 + i] - delta[subOffset0 + i] + delta[addOffset1 + i] - delta[subOffset1 + i];
    }
  }

  static inline auto add(std::span<Type, OutputCount> accumulator, std::span<const Type, WeightCount> delta, u32 offset) -> void {
    for (u32 i = 0; i < OutputCount; ++i) {
      accumulator[i] += delta[offset + i];
    }
  }

  static inline auto sub(std::span<Type, OutputCount> accumulator, std::span<const Type, WeightCount> delta, u32 offset) -> void {
    for (u32 i = 0; i < OutputCount; ++i) {
      accumulator[i] -= delta[offset + i];
    }
  }
};

template <typename Acc>
struct RefreshTableEntry {
  Acc accumulator{};
  std::array<BitboardSet, 2> bbs{};

  [[nodiscard]] auto colorBbs(chess::color c) -> auto & { return bbs[color_index(c)]; }
};

template <typename Ft, u32 BucketCount>
struct RefreshTable {
  std::array<RefreshTableEntry<Accumulator<Ft>>, BucketCount> table{};

  inline void init(const Ft &featureTransformer) {
    for (auto &entry : table) {
      entry.accumulator.initBoth(featureTransformer);
      entry.bbs.fill(BitboardSet{});
    }
  }
};

template <typename Type, u32 Inputs, u32 Outputs, typename FeatureSet = features::SingleBucket>
struct FeatureTransformer {
  using WeightType = Type;
  using OutputType = Type;

  using InputFeatureSet = FeatureSet;

  using Accumulator = Accumulator<FeatureTransformer<Type, Inputs, Outputs, FeatureSet>>;
  using RefreshTable = RefreshTable<FeatureTransformer<Type, Inputs, Outputs, FeatureSet>, FeatureSet::BucketCount>;

  static constexpr auto InputCount = InputFeatureSet::BucketCount * Inputs;
  static constexpr auto OutputCount = Outputs;

  static constexpr auto WeightCount = InputCount * OutputCount;
  static constexpr auto BiasCount = OutputCount;

  EVAL_SIMD_ALIGNAS std::array<WeightType, WeightCount> weights;
  EVAL_SIMD_ALIGNAS std::array<OutputType, BiasCount> biases;

  inline auto readFrom(IParamStream &stream) -> bool { return stream.read(weights) && stream.read(biases); }
  inline auto writeTo(IParamStream &stream) const -> bool { return stream.write(weights) && stream.write(biases); }
};
}  // namespace eval::nnue
