/*
  Atomic Syzygy backend for Seer
  (stub implementation – WDL/DTZ probing to be ported from mv_syzygy)
*/

#pragma once

#include <cstdint>
#include <string>

namespace chess {
class board;
}  // namespace chess

namespace search::atomic_tb {

enum class WDL : int8_t { Loss = -1, Draw = 0, Win = 1 };

struct ProbeResult {
  WDL wdl{WDL::Draw};
  int16_t dtz{-1};  // optional; -1 when unavailable
};

bool init(const std::string& path) noexcept;
bool probe_wdl(const chess::board& bd, ProbeResult& out) noexcept;
bool probe_dtz(const chess::board& bd, ProbeResult& out) noexcept;
void close() noexcept;

}  // namespace search::atomic_tb
