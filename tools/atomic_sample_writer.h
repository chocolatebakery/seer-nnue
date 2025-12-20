#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <chess/square.h>
#include <chess/types.h>
#include <sample.h>

namespace train {

struct atomic_sample_writer {
  std::string path_;
  std::ofstream file_;

  atomic_sample_writer& append_sample(const sample& datum) {
    struct entry {
      std::uint8_t code;
      std::uint8_t sq;
    };

    std::vector<entry> pieces{};
    pieces.reserve(static_cast<std::size_t>(datum.state_.num_pieces()));

    auto add_piece = [&](const chess::color side, const chess::piece_type pt, const chess::square& sq) {
      const auto base = static_cast<std::uint8_t>(pt);
      const std::uint8_t code = (side == chess::color::black) ? static_cast<std::uint8_t>(base + 6) : base;
      const int file = sq.file();
      const int rank = sq.rank();
      const std::uint8_t sq_idx = static_cast<std::uint8_t>(rank * 8 + (7 - file));
      pieces.push_back(entry{code, sq_idx});
    };

    chess::over_types([&](const chess::piece_type pt) {
      for (const auto sq : datum.state_.man_.white.get_plane(pt)) { add_piece(chess::color::white, pt, sq); }
      for (const auto sq : datum.state_.man_.black.get_plane(pt)) { add_piece(chess::color::black, pt, sq); }
    });

    std::sort(pieces.begin(), pieces.end(), [](const entry& a, const entry& b) {
      if (a.code != b.code) { return a.code < b.code; }
      return a.sq < b.sq;
    });

    const std::uint8_t n = static_cast<std::uint8_t>(pieces.size());
    const std::uint8_t stm = datum.state_.turn() ? 1 : 0;
    file_.write(reinterpret_cast<const char*>(&n), sizeof(n));
    file_.write(reinterpret_cast<const char*>(&stm), sizeof(stm));
    for (const auto& piece : pieces) {
      file_.write(reinterpret_cast<const char*>(&piece.code), sizeof(piece.code));
      file_.write(reinterpret_cast<const char*>(&piece.sq), sizeof(piece.sq));
    }

    const score_type min_score = static_cast<score_type>(std::numeric_limits<std::int16_t>::min());
    const score_type max_score = static_cast<score_type>(std::numeric_limits<std::int16_t>::max());
    const score_type clamped = std::clamp(datum.score_, min_score, max_score);
    const std::int16_t raw_score = static_cast<std::int16_t>(clamped);
    std::int8_t raw_result = 1;
    switch (datum.result_) {
      case result_type::loss: raw_result = 0; break;
      case result_type::draw: raw_result = 1; break;
      case result_type::win: raw_result = 2; break;
      default: raw_result = 1; break;
    }
    file_.write(reinterpret_cast<const char*>(&raw_score), sizeof(raw_score));
    file_.write(reinterpret_cast<const char*>(&raw_result), sizeof(raw_result));

    return *this;
  }

  explicit atomic_sample_writer(const std::string& path) : path_{path}, file_{path, std::ios::binary} {}
};

}  // namespace train
