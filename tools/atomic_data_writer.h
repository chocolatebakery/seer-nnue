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

#include <chess/board.h>
#include <search/search_constants.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include <marlinformat.h>

namespace train {

struct datagen_entry {
  chess::board state;
  search::score_type white_score{0};
};

struct atomic_data_writer {
  std::mutex writer_mutex_;
  std::ofstream file_;

  std::uint64_t total_;
  std::atomic_uint64_t completed_;

  std::uint64_t report_every_{0};
  std::uint64_t next_report_{0};
  std::chrono::steady_clock::time_point start_time_{};
  bool progress_enabled_{false};

  bool is_complete() {
    std::lock_guard<std::mutex> lck(writer_mutex_);
    return completed_ >= total_;
  }

  std::tuple<std::uint64_t, std::uint64_t> progress() const { return std::tuple(completed_.load(), total_); }

  void set_progress_every(const std::uint64_t report_every) {
    std::lock_guard<std::mutex> lck(writer_mutex_);
    report_every_ = report_every;
    progress_enabled_ = report_every_ > 0;
    if (progress_enabled_) {
      start_time_ = std::chrono::steady_clock::now();
      const auto completed = completed_.load();
      next_report_ = ((completed / report_every_) + 1) * report_every_;
    } else {
      next_report_ = 0;
    }
  }

  atomic_data_writer& write_block(const std::vector<datagen_entry>& data, const Outcome& outcome) {
    constexpr search::score_type nnue_scale_numerator = 1024;
    constexpr search::score_type nnue_scale_denominator = 288;

    std::lock_guard<std::mutex> lck(writer_mutex_);
    for (const auto& elem : data) {
      if (completed_ >= total_) { break; }

      const search::score_type cp_score = elem.white_score * nnue_scale_denominator / nnue_scale_numerator;
      const search::score_type min_score = static_cast<search::score_type>(std::numeric_limits<std::int16_t>::min());
      const search::score_type max_score = static_cast<search::score_type>(std::numeric_limits<std::int16_t>::max());
      const auto clamped = std::clamp(cp_score, min_score, max_score);
      const auto packed = marlinformat::PackedBoard::pack(elem.state, static_cast<std::int16_t>(clamped));

      marlinformat::PackedBoard out = packed;
      out.wdl = outcome;

      file_.write(reinterpret_cast<const char*>(&out), sizeof(out));
      ++completed_;
    }

    const auto completed = completed_.load();
    if (progress_enabled_ && report_every_ > 0 && completed >= next_report_) {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(now - start_time_).count();
      const double rate = elapsed > 0.0 ? static_cast<double>(completed) / elapsed : 0.0;
      const std::uint64_t rate_int = static_cast<std::uint64_t>(rate);
      const std::uint64_t percent = total_ > 0 ? (completed * 100ULL) / total_ : 0ULL;
      std::cerr << "progress " << completed << "/" << total_ << " (" << percent << "%) " << rate_int << " samples/s\n";
      std::cerr.flush();
      next_report_ = ((completed / report_every_) + 1) * report_every_;
    }

    return *this;
  }

  atomic_data_writer(const std::string& write_path, const std::size_t& total)
      : file_{write_path, std::ios::binary}, total_{total}, completed_{0} {}
};

}  // namespace train
