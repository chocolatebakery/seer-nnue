/*
  Atomic Syzygy core loader/decoder (WDL-only scaffold)
  This module owns table loading and low-level probing for Atomic TBs.
*/

#pragma once

#include <cstdint>
#include <string>

#include <search/atomic_tbprobe.h>  // for ProbeResult/WDL

namespace chess {
class board;
}  // namespace chess

namespace search::atomic_syzygy_core {

// Load all .atbw tables found under the given path. Returns true if at least one table is loaded.
bool init(const std::string& path) noexcept;

// Probe WDL for a given board. Returns true if a table was found and the probe succeeded.
bool probe_wdl(const chess::board& bd, atomic_tb::ProbeResult& out) noexcept;

// Cleanup any loaded resources.
void close() noexcept;

}  // namespace search::atomic_syzygy_core
