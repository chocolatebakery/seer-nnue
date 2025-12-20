/*
  Atomic Syzygy backend for Seer
  Current status: skeleton + board -> TBPosition conversion.
  WDL/DTZ probing will be ported from mv_syzygy/tbprobe.cpp.
*/

#include <search/atomic_tbprobe.h>

#include <algorithm>
#include <array>
#include <string>

#include <chess/board.h>
#include <chess/move.h>
#include <chess/square.h>
#include <search/atomic_syzygy_core.h>

// Internal representation of a small (<=6) Atomic TB position
enum TBPiece : uint8_t { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK };

struct TBPosition {
  uint8_t stm{0};   // 0 = black, 1 = white
  uint8_t n{0};
  TBPiece pc[6]{};
  uint8_t sq[6]{};
};

namespace {

// Convert Seer board to a canonical TB position (sorted by piece code then square).
inline bool board_to_tbpos(const chess::board& bd, TBPosition& out) {
  if (bd.num_pieces() > 6) { return false; }
  if (bd.man_.white.king().count() != 1 || bd.man_.black.king().count() != 1) { return false; }

  auto push = [&](chess::square_set mask, const TBPiece code, uint8_t& idx) -> bool {
    chess::square::data_type bb = mask.data;
    while (bb) {
      const int sq = __builtin_ctzll(bb);
      bb &= (bb - static_cast<chess::square::data_type>(1));
      if (idx >= 6) { return false; }  // defensive, though num_pieces() <= 6
      out.pc[idx] = code;
      out.sq[idx] = static_cast<uint8_t>(sq);
      ++idx;
    }
    return true;
  };

  uint8_t idx = 0;
  if (!push(bd.man_.white.pawn(), TBPiece::WP, idx)) { return false; }
  if (!push(bd.man_.white.knight(), TBPiece::WN, idx)) { return false; }
  if (!push(bd.man_.white.bishop(), TBPiece::WB, idx)) { return false; }
  if (!push(bd.man_.white.rook(), TBPiece::WR, idx)) { return false; }
  if (!push(bd.man_.white.queen(), TBPiece::WQ, idx)) { return false; }
  if (!push(bd.man_.white.king(), TBPiece::WK, idx)) { return false; }

  if (!push(bd.man_.black.pawn(), TBPiece::BP, idx)) { return false; }
  if (!push(bd.man_.black.knight(), TBPiece::BN, idx)) { return false; }
  if (!push(bd.man_.black.bishop(), TBPiece::BB, idx)) { return false; }
  if (!push(bd.man_.black.rook(), TBPiece::BR, idx)) { return false; }
  if (!push(bd.man_.black.queen(), TBPiece::BQ, idx)) { return false; }
  if (!push(bd.man_.black.king(), TBPiece::BK, idx)) { return false; }

  out.n = idx;
  if (out.n != bd.num_pieces()) { return false; }
  out.stm = bd.turn() ? 1 : 0;

  // Canonical ordering expected by the atomic Syzygy tables: sort by (piece code, square)
  std::array<std::pair<uint8_t, uint8_t>, 6> tmp{};
  for (uint8_t i = 0; i < idx; ++i) { tmp[i] = {out.pc[i], out.sq[i]}; }
  std::sort(tmp.begin(), tmp.begin() + idx);
  for (uint8_t i = 0; i < idx; ++i) {
    out.pc[i] = static_cast<TBPiece>(tmp[i].first);
    out.sq[i] = tmp[i].second;
  }

  return true;
}

}  // namespace

namespace search::atomic_tb {

bool init(const std::string& path) noexcept {
  return atomic_syzygy_core::init(path);
}

bool probe_wdl(const chess::board& bd, ProbeResult& out) noexcept {
  TBPosition pos{};
  if (!board_to_tbpos(bd, pos)) { return false; }

  // TODO: implementar probing real; stub por agora
  (void)pos;
  out.wdl = WDL::Draw;
  out.dtz = -1;
  return atomic_syzygy_core::probe_wdl(bd, out);
}

bool probe_dtz(const chess::board& bd, ProbeResult& out) noexcept {
  TBPosition pos{};
  if (!board_to_tbpos(bd, pos)) { return false; }

  // TODO: implementar probing DTZ real; stub por agora
  (void)pos;
  out.wdl = WDL::Draw;
  out.dtz = -1;
  return atomic_syzygy_core::probe_wdl(bd, out);
}

void close() noexcept {
  atomic_syzygy_core::close();
}

}  // namespace search::atomic_tb
