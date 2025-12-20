#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <tuple>

#include <atomic_sample_writer.h>

namespace train {

struct atomic_data_writer {
  std::mutex writer_mutex_;
  atomic_sample_writer writer_;

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

  std::tuple<std::uint64_t, std::uint64_t> progress() const {
    return std::tuple(completed_.load(), total_);
  }

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

  atomic_data_writer& write_block(const std::vector<sample>& data) {
    std::lock_guard<std::mutex> lck(writer_mutex_);
    for (const auto& elem : data) {
      if (completed_ >= total_) { break; }
      writer_.append_sample(elem);
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

  atomic_data_writer(const std::string& write_path, const size_t& total) : writer_{write_path}, total_{total}, completed_{0} {}
};

}  // namespace train
