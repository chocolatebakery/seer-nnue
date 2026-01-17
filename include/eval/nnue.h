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

#include "arch.h"
#include "types.h"

#include "nnue/activation.h"
#include "nnue/bitboard_set.h"
#include "nnue/coords.h"
#include "nnue/features.h"
#include "nnue/input.h"
#include "nnue/layers.h"
#include "nnue/network.h"
#include "util/static_vector.h"

#include <chess/board.h>
#include <chess/square.h>
#include <chess/types.h>

#include <array>
#include <cassert>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace eval {
using FeatureTransformer = nnue::FeatureTransformer<i16, InputSize, Layer1Size, InputFeatureSet>;

using Network = nnue::PerspectiveNetwork<
    FeatureTransformer,
    nnue::DensePerspectiveAffineLayer<i16, i16, L1Activation, Layer1Size, 1, OutputBucketing>>;

using Accumulator = FeatureTransformer::Accumulator;
using RefreshTable = FeatureTransformer::RefreshTable;

extern const Network &g_network;

auto loadDefaultNetwork() -> void;
auto loadNetwork(const std::string &name) -> void;
[[nodiscard]] auto defaultNetworkName() -> std::string_view;
[[nodiscard]] auto loadedNetworkName() -> std::string_view;

struct NnueUpdates {
  struct PieceSquare {
    chess::color color{chess::color::white};
    chess::piece_type piece{chess::piece_type::pawn};
    chess::square square{chess::square::from_index(0)};
  };

  static constexpr eval::usize MaxSubs = 16;
  static constexpr eval::usize MaxAdds = 4;

  std::array<bool, 2> refresh{};
  util::StaticVector<PieceSquare, MaxSubs> sub{};
  util::StaticVector<PieceSquare, MaxAdds> add{};

  inline auto setRefresh(chess::color c) { refresh[nnue::color_index(c)] = true; }

  inline auto pushSubAdd(chess::color c, chess::piece_type piece, chess::square src, chess::square dst) {
    sub.push(PieceSquare{c, piece, src});
    add.push(PieceSquare{c, piece, dst});
  }

  inline auto pushSub(chess::color c, chess::piece_type piece, chess::square square) {
    sub.push(PieceSquare{c, piece, square});
  }

  inline auto pushAdd(chess::color c, chess::piece_type piece, chess::square square) {
    add.push(PieceSquare{c, piece, square});
  }
};

class NnueState {
 public:
  NnueState() { accumulatorStack_.resize(256); }

  inline auto reset(const chess::board &bd) {
    assert(bd.man_.black.king().any());
    assert(bd.man_.white.king().any());
    const auto blackKing = bd.man_.black.king().item();
    const auto whiteKing = bd.man_.white.king().item();

    const auto bbs = nnue::BitboardSet::from_board(bd);

    refreshTable_.init(g_network.featureTransformer());
    curr_ = &accumulatorStack_[0];

    for (const auto c : {chess::color::black, chess::color::white}) {
      const auto king = c == chess::color::black ? blackKing : whiteKing;
      const auto bucket = InputFeatureSet::getBucket(c, king);

      auto &rtEntry = refreshTable_.table[bucket];
      resetAccumulator(rtEntry.accumulator, c, bbs, king);

      curr_->copyFrom(c, rtEntry.accumulator);
      rtEntry.colorBbs(c) = bbs;
    }
  }

  template <bool Push>
  inline auto update(const NnueUpdates &updates, const chess::board &bd) {
    assert(curr_ != nullptr);
    auto *next = Push ? curr_ + 1 : curr_;
    assert(next <= &accumulatorStack_.back());

    const auto bbs = nnue::BitboardSet::from_board(bd);
    const auto blackKing = bd.man_.black.king().item();
    const auto whiteKing = bd.man_.white.king().item();

    const auto subCount = updates.sub.size();
    const auto addCount = updates.add.size();
    assert(subCount <= NnueUpdates::MaxSubs);
    assert(addCount <= NnueUpdates::MaxAdds);

    for (const auto c : {chess::color::black, chess::color::white}) {
      const auto king = c == chess::color::black ? blackKing : whiteKing;

      if (updates.refresh[nnue::color_index(c)]) {
        refreshAccumulator(*next, c, bbs, refreshTable_, king);
        continue;
      }

      if (addCount == 0 && subCount == 0) {
        next->copyFrom(c, *curr_);
        continue;
      }

      if (addCount == 0 && subCount > 0) {
        std::array<u32, NnueUpdates::MaxSubs> subs{};
        for (eval::usize i = 0; i < subCount; ++i) {
          const auto &entry = updates.sub[i];
          subs[i] = featureIndex(c, entry.color, entry.piece, entry.square, king);
        }
        next->subFrom(*curr_, g_network.featureTransformer(), c, std::span<const u32>{subs.data(), subCount});
        continue;
      }

      if (addCount == 1 && subCount == 1) {
        const auto &subEntry = updates.sub[0];
        const auto &addEntry = updates.add[0];
        const auto sub = featureIndex(c, subEntry.color, subEntry.piece, subEntry.square, king);
        const auto add = featureIndex(c, addEntry.color, addEntry.piece, addEntry.square, king);
        next->subAddFrom(*curr_, g_network.featureTransformer(), c, sub, add);
        continue;
      }

      if (addCount == 1 && subCount == 2) {
        const auto &subEntry0 = updates.sub[0];
        const auto &subEntry1 = updates.sub[1];
        const auto &addEntry = updates.add[0];

        const auto sub0 = featureIndex(c, subEntry0.color, subEntry0.piece, subEntry0.square, king);
        const auto sub1 = featureIndex(c, subEntry1.color, subEntry1.piece, subEntry1.square, king);
        const auto add = featureIndex(c, addEntry.color, addEntry.piece, addEntry.square, king);

        next->subSubAddFrom(*curr_, g_network.featureTransformer(), c, sub0, sub1, add);
        continue;
      }

      if (addCount == 2 && subCount == 2) {
        const auto &subEntry0 = updates.sub[0];
        const auto &subEntry1 = updates.sub[1];
        const auto &addEntry0 = updates.add[0];
        const auto &addEntry1 = updates.add[1];

        const auto sub0 = featureIndex(c, subEntry0.color, subEntry0.piece, subEntry0.square, king);
        const auto sub1 = featureIndex(c, subEntry1.color, subEntry1.piece, subEntry1.square, king);
        const auto add0 = featureIndex(c, addEntry0.color, addEntry0.piece, addEntry0.square, king);
        const auto add1 = featureIndex(c, addEntry1.color, addEntry1.piece, addEntry1.square, king);

        next->subSubAddAddFrom(*curr_, g_network.featureTransformer(), c, sub0, sub1, add0, add1);
        continue;
      }

      next->copyFrom(c, *curr_);
      for (eval::usize i = 0; i < subCount; ++i) {
        const auto &entry = updates.sub[i];
        const auto idx = featureIndex(c, entry.color, entry.piece, entry.square, king);
        next->deactivateFeature(g_network.featureTransformer(), c, idx);
      }
      for (eval::usize i = 0; i < addCount; ++i) {
        const auto &entry = updates.add[i];
        const auto idx = featureIndex(c, entry.color, entry.piece, entry.square, king);
        next->activateFeature(g_network.featureTransformer(), c, idx);
      }
    }

    curr_ = next;
  }

  inline auto pop() {
    if (curr_ > &accumulatorStack_[0]) {
      --curr_;
    }
  }

  [[nodiscard]] inline auto evaluate(const chess::board &bd, chess::color stm) const -> i32 {
    assert(curr_ != nullptr);
    const auto bbs = nnue::BitboardSet::from_board(bd);
    return evaluate(*curr_, bbs, stm);
  }

  [[nodiscard]] inline auto outputs(chess::color c) const
      -> std::span<const FeatureTransformer::OutputType, FeatureTransformer::OutputCount> {
    return curr_->forColor(c);
  }

  [[nodiscard]] static inline auto evaluateOnce(const chess::board &bd, chess::color stm) -> i32 {
    assert(bd.man_.black.king().any());
    assert(bd.man_.white.king().any());
    Accumulator accumulator{};
    const auto bbs = nnue::BitboardSet::from_board(bd);

    accumulator.initBoth(g_network.featureTransformer());
    resetAccumulator(accumulator, chess::color::black, bbs, bd.man_.black.king().item());
    resetAccumulator(accumulator, chess::color::white, bbs, bd.man_.white.king().item());

    return evaluate(accumulator, bbs, stm);
  }

 private:
  std::vector<Accumulator> accumulatorStack_{};
  Accumulator *curr_{};
  RefreshTable refreshTable_{};

  [[nodiscard]] static inline auto evaluate(const Accumulator &accumulator, const nnue::BitboardSet &bbs, chess::color stm) -> i32 {
    constexpr i32 Q = L1Q * OutputQ;
    const auto output = stm == chess::color::black
        ? g_network.propagate(bbs, accumulator.black(), accumulator.white())
        : g_network.propagate(bbs, accumulator.white(), accumulator.black());
    return static_cast<i32>(output) * Scale / Q;
  }

  static inline auto refreshAccumulator(Accumulator &accumulator, chess::color c, const nnue::BitboardSet &bbs,
      RefreshTable &refreshTable, chess::square king) -> void {
    const auto bucket = InputFeatureSet::getBucket(c, king);
    auto &rtEntry = refreshTable.table[bucket];
    auto &prevBoards = rtEntry.colorBbs(c);

    for (const auto pieceColor : {chess::color::black, chess::color::white}) {
      chess::over_types([&](const chess::piece_type pt) {
        const auto prev = prevBoards.forPiece(pt, pieceColor);
        const auto curr = bbs.forPiece(pt, pieceColor);

        const auto added = curr & ~prev;
        const auto removed = prev & ~curr;

        for (const auto sq : added) {
          const auto feature = featureIndex(c, pieceColor, pt, sq, king);
          rtEntry.accumulator.activateFeature(g_network.featureTransformer(), c, feature);
        }

        for (const auto sq : removed) {
          const auto feature = featureIndex(c, pieceColor, pt, sq, king);
          rtEntry.accumulator.deactivateFeature(g_network.featureTransformer(), c, feature);
        }
      });
    }

    accumulator.copyFrom(c, rtEntry.accumulator);
    prevBoards = bbs;
  }

  static inline auto resetAccumulator(Accumulator &accumulator, chess::color c, const nnue::BitboardSet &bbs,
      chess::square king) -> void {
    for (const auto pieceColor : {chess::color::black, chess::color::white}) {
      chess::over_types([&](const chess::piece_type pt) {
        const auto board = bbs.forPiece(pt, pieceColor);
        for (const auto sq : board) {
          const auto feature = featureIndex(c, pieceColor, pt, sq, king);
          accumulator.activateFeature(g_network.featureTransformer(), c, feature);
        }
      });
    }
  }

  [[nodiscard]] static inline auto featureIndex(
      chess::color perspective, chess::color pieceColor, chess::piece_type piece, chess::square sq, chess::square king) -> u32 {
    constexpr u32 ColorStride = 64U * 6U;
    constexpr u32 PieceStride = 64U;

    const auto type = static_cast<u32>(piece);
    const u32 color = pieceColor == perspective ? 0U : 1U;
    const auto bucketOffset = InputFeatureSet::getBucket(perspective, king) * InputSize;
    return bucketOffset + color * ColorStride + type * PieceStride + nnue::feature_square_index(sq, perspective);
  }
};

[[nodiscard]] inline auto buildUpdates(const chess::board &before, const chess::board &after) -> NnueUpdates {
  NnueUpdates updates{};

  const auto prevBlackKing = before.man_.black.king().item();
  const auto prevWhiteKing = before.man_.white.king().item();
  const auto nextBlackKing = after.man_.black.king().item();
  const auto nextWhiteKing = after.man_.white.king().item();

  if (InputFeatureSet::refreshRequired(chess::color::black, prevBlackKing, nextBlackKing)) {
    updates.setRefresh(chess::color::black);
  }
  if (InputFeatureSet::refreshRequired(chess::color::white, prevWhiteKing, nextWhiteKing)) {
    updates.setRefresh(chess::color::white);
  }

  for (const auto c : {chess::color::white, chess::color::black}) {
    const auto &beforeConfig = before.man_.us(c);
    const auto &afterConfig = after.man_.us(c);

    chess::over_types([&](const chess::piece_type pt) {
      const auto beforePlane = beforeConfig.get_plane(pt);
      const auto afterPlane = afterConfig.get_plane(pt);

      const auto removed = beforePlane & ~afterPlane;
      for (const auto sq : removed) {
        updates.pushSub(c, pt, sq);
      }

      const auto added = afterPlane & ~beforePlane;
      for (const auto sq : added) {
        updates.pushAdd(c, pt, sq);
      }
    });
  }

  return updates;
}
}  // namespace eval
