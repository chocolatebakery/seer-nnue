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

#include <chess/board.h>
#include <chess/castle_info.h>
#include <chess/cuckoo_hash_table.h>
#include <chess/pawn_info.h>
#include <chess/table_generation.h>

#include <array>
#include <cstdint>
#include <limits>
#include <sstream>

namespace chess {

namespace {

inline square_set explosion_mask(const square& center) noexcept {
  return king_attack_tbl.look_up(center) | square_set(center.bit_board());
}

inline piece_type piece_at_unchecked(const board& bd, const square& sq) noexcept {
  return bd.man_.white.all().is_member(sq) ? bd.man_.white.occ(sq) : bd.man_.black.occ(sq);
}

template <color attacker>
inline square_set attack_to(const board& bd, const square& tgt, const square_set& occ) noexcept {
  square_set result{};
  // use opposite color lookup so we get squares from which an attacker pawn
  // could capture onto tgt
  result |= pawn_attack_tbl<opponent<attacker>>.look_up(tgt) & bd.man_.us<attacker>().pawn();
  result |= knight_attack_tbl.look_up(tgt) & bd.man_.us<attacker>().knight();
  const square_set diag = bishop_attack_tbl.look_up(tgt, occ);
  const square_set ortho = rook_attack_tbl.look_up(tgt, occ);
  result |= diag & (bd.man_.us<attacker>().bishop() | bd.man_.us<attacker>().queen());
  result |= ortho & (bd.man_.us<attacker>().rook() | bd.man_.us<attacker>().queen());
  // king captures are illegal in atomic, so we skip king attackers
  return result;
}

template <color side>
inline square_set king_ring(const board& bd) noexcept {
  const square_set king_bb = bd.man_.us<side>().king();
  if (!king_bb.any()) { return square_set{}; }
  return king_attack_tbl.look_up(king_bb.item());
}

template <color attacker>
inline bool immediate_indirect_kill(const board& bd, const square_set& occ_after, const square_set& our_ring) noexcept {
  for (const auto r : (our_ring & occ_after)) {
    if (attack_to<attacker>(bd, r, occ_after).any()) { return true; }
  }
  return false;
}

template <color attacker>
inline bool king_capturable_in_position(const board& bd) noexcept {
  constexpr color defender = opponent<attacker>;

  const square_set defender_king_set = bd.man_.us<defender>().king();
  if (!defender_king_set.any()) { return false; }

  const square_set occ = bd.man_.white.all() | bd.man_.black.all();

  auto good_capture = [&](const board& after) {
    const bool defender_alive = after.man_.us<defender>().king().any();
    const bool attacker_alive = after.man_.us<attacker>().king().any();
    return (!defender_alive) && attacker_alive;
  };

  // pawns
  for (const auto from : bd.man_.us<attacker>().pawn()) {
    const square_set targets = pawn_attack_tbl<attacker>.look_up(from) & bd.man_.us<defender>().all();
    for (const auto to : targets) {
      const piece_type captured = bd.man_.us<defender>().occ(to);
      const move mv(from, to, piece_type::pawn, true, captured);
      const board after = bd.forward_<attacker>(mv);
      if (good_capture(after)) { return true; }
    }
  }

  // en passant
  if (const square_set ep_sq_set = bd.lat_.them<attacker>().ep_mask(); ep_sq_set.any()) {
    const square ep_sq = ep_sq_set.item();
    const square cap_sq = pawn_push_tbl<opponent<attacker>>.look_up(ep_sq, square_set{}).item();
    const square_set from_mask = pawn_attack_tbl<attacker>.look_up(ep_sq) & bd.man_.us<attacker>().pawn();
    for (const auto from : from_mask) {
      const move mv(from, ep_sq, piece_type::pawn, true, piece_type::pawn, true, cap_sq);
      const board after = bd.forward_<attacker>(mv);
      if (good_capture(after)) { return true; }
    }
  }

  // knights
  for (const auto from : bd.man_.us<attacker>().knight()) {
    const square_set targets = knight_attack_tbl.look_up(from) & bd.man_.us<defender>().all();
    for (const auto to : targets) {
      const piece_type captured = bd.man_.us<defender>().occ(to);
      const move mv(from, to, piece_type::knight, true, captured);
      const board after = bd.forward_<attacker>(mv);
      if (good_capture(after)) { return true; }
    }
  }

  // bishops
  for (const auto from : bd.man_.us<attacker>().bishop()) {
    const square_set targets = bishop_attack_tbl.look_up(from, occ) & bd.man_.us<defender>().all();
    for (const auto to : targets) {
      const piece_type captured = bd.man_.us<defender>().occ(to);
      const move mv(from, to, piece_type::bishop, true, captured);
      const board after = bd.forward_<attacker>(mv);
      if (good_capture(after)) { return true; }
    }
  }

  // rooks
  for (const auto from : bd.man_.us<attacker>().rook()) {
    const square_set targets = rook_attack_tbl.look_up(from, occ) & bd.man_.us<defender>().all();
    for (const auto to : targets) {
      const piece_type captured = bd.man_.us<defender>().occ(to);
      const move mv(from, to, piece_type::rook, true, captured);
      const board after = bd.forward_<attacker>(mv);
      if (good_capture(after)) { return true; }
    }
  }

  // queens
  for (const auto from : bd.man_.us<attacker>().queen()) {
    const square_set targets = (bishop_attack_tbl.look_up(from, occ) | rook_attack_tbl.look_up(from, occ)) & bd.man_.us<defender>().all();
    for (const auto to : targets) {
      const piece_type captured = bd.man_.us<defender>().occ(to);
      const move mv(from, to, piece_type::queen, true, captured);
      const board after = bd.forward_<attacker>(mv);
      if (good_capture(after)) { return true; }
    }
  }

  return false;
}

}  // namespace

square_set board::capture_blast(const square& center) noexcept { return king_attack_tbl.look_up(center) | square_set(center.bit_board()); }

bool board::is_atomic_king_blast_capture(const move& mv) const noexcept {
  if (!(mv.is_capture() || mv.is_enpassant())) { return false; }

  // For en passant, explosion is centered on the captured pawn's square, not mv.to()
  const square center = [&] {
    if (mv.is_enpassant()) {
      // Calculate the square of the captured pawn (one rank back from destination)
      return turn()
        ? pawn_push_tbl<color::black>.look_up(mv.to(), square_set{}).item()
        : pawn_push_tbl<color::white>.look_up(mv.to(), square_set{}).item();
    }
    return mv.to();
  }();

  const square_set blast = capture_blast(center);
  return (blast & man_.them(turn()).king()).any();
}

template <color c>
inline bool has_atomic_blast_capture_(const board& bd) noexcept {
  const square_set enemy_king = bd.man_.them<c>().king();
  if (!enemy_king.any()) { return false; }
  const square_set king_zone = explosion_mask(enemy_king.item());
  square_set targets = king_zone & bd.man_.them<c>().all();
  const square_set our_king = bd.man_.us<c>().king();
  const square_set occ = bd.man_.white.all() | bd.man_.black.all();

  while (targets.any()) {
    const square t = targets.item();
    targets = square_set(targets.data & (targets.data - static_cast<square::data_type>(1)));
    if ((explosion_mask(t) & our_king).any()) { continue; }
    if (attack_to<c>(bd, t, occ).any()) { return true; }
  }
  return false;
}

bool board::has_atomic_blast_capture() const noexcept {
  return turn() ? has_atomic_blast_capture_<color::white>(*this) : has_atomic_blast_capture_<color::black>(*this);
}

bool board::has_atomic_blast_capture_for(const color side) const noexcept {
  return side == color::white ? has_atomic_blast_capture_<color::white>(*this) : has_atomic_blast_capture_<color::black>(*this);
}

bool board::in_atomic_blast_check() const noexcept {
  const color them = turn() ? color::black : color::white;
  return has_atomic_blast_capture_for(them);
}

template <typename T>
constexpr T material_value(const piece_type& pt) {
  constexpr std::array<T, 6> values = {100, 300, 300, 450, 900, std::numeric_limits<T>::max()};
  return values[static_cast<std::size_t>(pt)];
}

template <typename T>
constexpr T phase_value(const piece_type& pt) {
  constexpr std::array<T, 6> values = {0, 1, 1, 2, 4, 0};
  return values[static_cast<std::size_t>(pt)];
}

template <color c>
std::tuple<piece_type, square> board::least_valuable_attacker(const square& tgt, const square_set& ignore) const noexcept {
  const square_set p_mask = pawn_attack_tbl<opponent<c>>.look_up(tgt);
  const square_set p_attackers = p_mask & man_.us<c>().pawn() & ~ignore;
  if (p_attackers.any()) { return std::tuple(piece_type::pawn, *p_attackers.begin()); }

  const square_set n_mask = knight_attack_tbl.look_up(tgt);
  const square_set n_attackers = n_mask & man_.us<c>().knight() & ~ignore;
  if (n_attackers.any()) { return std::tuple(piece_type::knight, *n_attackers.begin()); }

  const square_set occ = (man_.white.all() | man_.black.all()) & ~ignore;

  const square_set b_mask = bishop_attack_tbl.look_up(tgt, occ);
  const square_set b_attackers = b_mask & man_.us<c>().bishop() & ~ignore;
  if (b_attackers.any()) { return std::tuple(piece_type::bishop, *b_attackers.begin()); }

  const square_set r_mask = rook_attack_tbl.look_up(tgt, occ);
  const square_set r_attackers = r_mask & man_.us<c>().rook() & ~ignore;
  if (r_attackers.any()) { return std::tuple(piece_type::rook, *r_attackers.begin()); }

  const square_set q_mask = b_mask | r_mask;
  const square_set q_attackers = q_mask & man_.us<c>().queen() & ~ignore;
  if (q_attackers.any()) { return std::tuple(piece_type::queen, *q_attackers.begin()); }

  const square_set k_mask = king_attack_tbl.look_up(tgt);
  const square_set k_attackers = k_mask & man_.us<c>().king() & ~ignore;
  if (k_attackers.any()) { return std::tuple(piece_type::king, *k_attackers.begin()); }

  return std::tuple(piece_type::pawn, tgt);
}

template <color c>
inline std::tuple<square_set, square_set> board::checkers(const square_set& occ) const noexcept {
  if (!man_.us<c>().king().any()) { return std::tuple(square_set{}, square_set{}); }
  const square_set b_check_mask = bishop_attack_tbl.look_up(man_.us<c>().king().item(), occ);
  const square_set r_check_mask = rook_attack_tbl.look_up(man_.us<c>().king().item(), occ);
  const square_set n_check_mask = knight_attack_tbl.look_up(man_.us<c>().king().item());
  // squares from which opponent pawns could capture our king (inverse uses our color)
  const square_set p_check_mask = pawn_attack_tbl<c>.look_up(man_.us<c>().king().item());
  const square_set q_check_mask = b_check_mask | r_check_mask;

  const square_set b_checkers = (b_check_mask & (man_.them<c>().bishop() | man_.them<c>().queen()));
  const square_set r_checkers = (r_check_mask & (man_.them<c>().rook() | man_.them<c>().queen()));

  square_set checker_rays_{};
  for (const auto sq : b_checkers) { checker_rays_ |= bishop_attack_tbl.look_up(sq, occ) & b_check_mask; }
  for (const auto sq : r_checkers) { checker_rays_ |= rook_attack_tbl.look_up(sq, occ) & r_check_mask; }

  const auto checkers_ = (b_check_mask & man_.them<c>().bishop() & occ) | (r_check_mask & man_.them<c>().rook() & occ) |
                         (n_check_mask & man_.them<c>().knight() & occ) | (p_check_mask & man_.them<c>().pawn() & occ) |
                         (q_check_mask & man_.them<c>().queen() & occ);
  return std::tuple(checkers_, checker_rays_);
}

template <color c>
inline square_set board::threat_mask() const noexcept {
  // idea from koivisto
  const square_set occ = man_.white.all() | man_.black.all();

  square_set threats{};
  square_set vulnerable = man_.them<c>().all();

  vulnerable &= ~man_.them<c>().pawn();
  square_set pawn_attacks{};
  for (const auto sq : man_.us<c>().pawn()) { pawn_attacks |= pawn_attack_tbl<c>.look_up(sq); }
  threats |= pawn_attacks & vulnerable;

  vulnerable &= ~(man_.them<c>().knight() | man_.them<c>().bishop());
  square_set minor_attacks{};
  for (const auto sq : man_.us<c>().knight()) { minor_attacks |= knight_attack_tbl.look_up(sq); }
  for (const auto sq : man_.us<c>().bishop()) { minor_attacks |= bishop_attack_tbl.look_up(sq, occ); }
  threats |= minor_attacks & vulnerable;

  vulnerable &= ~man_.them<c>().rook();
  square_set rook_attacks{};
  for (const auto sq : man_.us<c>().rook()) { rook_attacks |= rook_attack_tbl.look_up(sq, occ); }
  threats |= rook_attacks & vulnerable;

  return threats;
}

square_set board::us_threat_mask() const noexcept { return turn() ? threat_mask<color::white>() : threat_mask<color::black>(); }
square_set board::them_threat_mask() const noexcept { return turn() ? threat_mask<color::black>() : threat_mask<color::white>(); }

template <color c>
inline bool board::creates_threat_(const move& mv) const noexcept {
  const square_set occ = man_.white.all() | man_.black.all();
  auto attacks = [&occ](const piece_type& piece, const square& sq) {
    switch (piece) {
      case piece_type::pawn: return pawn_attack_tbl<c>.look_up(sq);
      case piece_type::knight: return knight_attack_tbl.look_up(sq);
      case piece_type::bishop: return bishop_attack_tbl.look_up(sq, occ);
      case piece_type::rook: return rook_attack_tbl.look_up(sq, occ);
      default: return square_set{};
    }
  };

  const square_set current_attacks = attacks(mv.piece(), mv.from());
  const square_set next_attacks = attacks(mv.piece(), mv.to());
  const square_set new_attacks = next_attacks & ~current_attacks;

  const square_set vulnerable = [&, this] {
    switch (mv.piece()) {
      case piece_type::pawn: return man_.them<c>().all() & ~(man_.them<c>().pawn() | man_.them<c>().king());
      case piece_type::knight: return man_.them<c>().rook() | man_.them<c>().queen();
      case piece_type::bishop: return man_.them<c>().rook() | man_.them<c>().queen();
      case piece_type::rook: return man_.them<c>().queen();
      default: return square_set{};
    }
  }();

  return (new_attacks & vulnerable).any();
}

bool board::creates_threat(const move& mv) const noexcept { return turn() ? creates_threat_<color::white>(mv) : creates_threat_<color::black>(mv); }

template <color c>
inline square_set board::king_danger() const noexcept {
  if (!man_.us<c>().king().any()) { return square_set{}; }
  const square_set occ = (man_.white.all() | man_.black.all()) & ~man_.us<c>().king();
  square_set k_danger{};
  for (const auto sq : man_.them<c>().pawn()) { k_danger |= pawn_attack_tbl<opponent<c>>.look_up(sq); }
  for (const auto sq : man_.them<c>().knight()) { k_danger |= knight_attack_tbl.look_up(sq); }
  for (const auto sq : man_.them<c>().king()) { k_danger |= king_attack_tbl.look_up(sq); }
  for (const auto sq : man_.them<c>().rook()) { k_danger |= rook_attack_tbl.look_up(sq, occ); }
  for (const auto sq : man_.them<c>().bishop()) { k_danger |= bishop_attack_tbl.look_up(sq, occ); }
  for (const auto sq : man_.them<c>().queen()) {
    k_danger |= rook_attack_tbl.look_up(sq, occ);
    k_danger |= bishop_attack_tbl.look_up(sq, occ);
  }
  return k_danger;
}

template <color c>
inline square_set board::pinned() const noexcept {
  if (!man_.us<c>().king().any()) { return square_set{}; }
  const square_set occ = man_.white.all() | man_.black.all();
  const auto k_x_diag = bishop_attack_tbl.look_up(man_.us<c>().king().item(), square_set{});
  const auto k_x_hori = rook_attack_tbl.look_up(man_.us<c>().king().item(), square_set{});
  const auto b_check_mask = bishop_attack_tbl.look_up(man_.us<c>().king().item(), occ);
  const auto r_check_mask = rook_attack_tbl.look_up(man_.us<c>().king().item(), occ);
  square_set pinned_set{};
  for (const auto sq : (k_x_hori & (man_.them<c>().queen() | man_.them<c>().rook()))) {
    pinned_set |= r_check_mask & rook_attack_tbl.look_up(sq, occ) & man_.us<c>().all();
  }
  for (const auto sq : (k_x_diag & (man_.them<c>().queen() | man_.them<c>().bishop()))) {
    pinned_set |= b_check_mask & bishop_attack_tbl.look_up(sq, occ) & man_.us<c>().all();
  }
  return pinned_set;
}

template <color c, typename mode>
inline void board::add_en_passant(move_list& mv_ls) const noexcept {
  if constexpr (!mode::noisy) { return; }
  if (lat_.them<c>().ep_mask().any()) {
    const square_set occ = man_.white.all() | man_.black.all();
    const square ep_square = lat_.them<c>().ep_mask().item();
    const square_set enemy_pawn_mask = pawn_push_tbl<opponent<c>>.look_up(ep_square, square_set{});
    const square_set from_mask = pawn_attack_tbl<opponent<c>>.look_up(ep_square) & man_.us<c>().pawn();
    for (const auto from : from_mask) {
      const square_set occ_ = (occ & ~square_set{from.bit_board()} & ~enemy_pawn_mask) | lat_.them<c>().ep_mask();
      if (!std::get<0>(checkers<c>(occ_)).any()) {
        mv_ls.push(from, ep_square, piece_type::pawn, false, piece_type::pawn, true, enemy_pawn_mask.item());
      }
    }
  }
}

template <color c, typename mode>
inline void board::add_castle(const move_generator_info& info, move_list& result) const noexcept {
  if constexpr (!mode::noisy) { return; }
  if (lat_.us<c>().oo() && !(castle_info<c>.oo_mask & (info.king_danger | info.occ)).any()) {
    result.push(castle_info<c>.start_king, castle_info<c>.oo_rook, piece_type::king, true, piece_type::rook);
  }
  if (lat_.us<c>().ooo() && !(castle_info<c>.ooo_danger_mask & info.king_danger).any() && !(castle_info<c>.ooo_occ_mask & info.occ).any()) {
    result.push(castle_info<c>.start_king, castle_info<c>.ooo_rook, piece_type::king, true, piece_type::rook);
  }
}

template <color c, typename mode>
inline void board::add_normal_pawn(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().pawn() & ~info.pinned)) {
    const auto to_quiet = pawn_push_tbl<c>.look_up(from, info.occ);
    const auto to_noisy = pawn_attack_tbl<c>.look_up(from) & man_.them<c>().all();
    if constexpr (mode::quiet) {
      for (const auto to : (to_quiet & ~info.last_rank)) { result.push(from, to, piece_type::pawn); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_noisy & ~info.last_rank)) { result.push(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
    }
    for (const auto to : (to_quiet & info.last_rank)) {
      if constexpr (mode::quiet) { result.push_under_promotions(from, to, piece_type::pawn); }
      if constexpr (mode::noisy) { result.push_queen_promotion(from, to, piece_type::pawn); }
    }
    for (const auto to : (to_noisy & info.last_rank)) {
      // for historical reasons, underpromotion captures are considered quiet
      if constexpr (mode::quiet) { result.push_under_promotions(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
      if constexpr (mode::noisy) { result.push_queen_promotion(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_normal_knight(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().knight() & ~info.pinned)) {
    const auto to_mask = knight_attack_tbl.look_up(from);
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.occ)) { result.push(from, to, piece_type::knight); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & man_.them<c>().all())) { result.push(from, to, piece_type::knight, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_normal_bishop(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().bishop() & ~info.pinned)) {
    const auto to_mask = bishop_attack_tbl.look_up(from, info.occ);
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.occ)) { result.push(from, to, piece_type::bishop); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & man_.them<c>().all())) { result.push(from, to, piece_type::bishop, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_normal_rook(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().rook() & ~info.pinned)) {
    const auto to_mask = rook_attack_tbl.look_up(from, info.occ);
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.occ)) { result.push(from, to, piece_type::rook); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & man_.them<c>().all())) { result.push(from, to, piece_type::rook, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_normal_queen(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().queen() & ~info.pinned)) {
    const auto to_mask = bishop_attack_tbl.look_up(from, info.occ) | rook_attack_tbl.look_up(from, info.occ);
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.occ)) { result.push(from, to, piece_type::queen); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & man_.them<c>().all())) { result.push(from, to, piece_type::queen, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_pinned_pawn(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().pawn() & info.pinned & info.king_diagonal)) {
    const auto to_mask = pawn_attack_tbl<c>.look_up(from) & info.king_diagonal;
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & ~info.last_rank & man_.them<c>().all())) {
        result.push(from, to, piece_type::pawn, true, man_.them<c>().occ(to));
      }
    }
    for (const auto to : (to_mask & info.last_rank & man_.them<c>().all())) {
      if constexpr (mode::quiet) { result.push_under_promotions(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
      if constexpr (mode::noisy) { result.push_queen_promotion(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
    }
  }
  for (const auto from : (man_.us<c>().pawn() & info.pinned & info.king_horizontal)) {
    const auto to_mask = pawn_push_tbl<c>.look_up(from, info.occ) & info.king_horizontal;
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.last_rank)) { result.push(from, to, piece_type::pawn); }
    }
    for (const auto to : (to_mask & info.last_rank)) {
      if constexpr (mode::quiet) { result.push_under_promotions(from, to, piece_type::pawn); }
      if constexpr (mode::noisy) { result.push_queen_promotion(from, to, piece_type::pawn); }
    }
  }
}

template <color c, typename mode>
inline void board::add_pinned_bishop(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().bishop() & info.pinned & info.king_diagonal)) {
    const auto to_mask = bishop_attack_tbl.look_up(from, info.occ) & info.king_diagonal;
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.occ)) { result.push(from, to, piece_type::bishop); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & man_.them<c>().all())) { result.push(from, to, piece_type::bishop, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_pinned_rook(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().rook() & info.pinned & info.king_horizontal)) {
    const auto to_mask = rook_attack_tbl.look_up(from, info.occ) & info.king_horizontal;
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.occ)) { result.push(from, to, piece_type::rook); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & man_.them<c>().all())) { result.push(from, to, piece_type::rook, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_pinned_queen(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().queen() & info.pinned & info.king_diagonal)) {
    const auto to_mask = bishop_attack_tbl.look_up(from, info.occ) & info.king_diagonal;
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.occ)) { result.push(from, to, piece_type::queen); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & man_.them<c>().all())) { result.push(from, to, piece_type::queen, true, man_.them<c>().occ(to)); }
    }
  }
  for (const auto from : (man_.us<c>().queen() & info.pinned & info.king_horizontal)) {
    const auto to_mask = rook_attack_tbl.look_up(from, info.occ) & info.king_horizontal;
    if constexpr (mode::quiet) {
      for (const auto to : (to_mask & ~info.occ)) { result.push(from, to, piece_type::queen); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_mask & man_.them<c>().all())) { result.push(from, to, piece_type::queen, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_checked_pawn(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().pawn() & ~info.pinned)) {
    const auto to_quiet = info.checker_rays & pawn_push_tbl<c>.look_up(from, info.occ);
    const auto to_noisy = info.checkers & pawn_attack_tbl<c>.look_up(from);
    if constexpr (mode::check) {
      for (const auto to : (to_quiet & ~info.last_rank)) { result.push(from, to, piece_type::pawn); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : (to_noisy & ~info.last_rank)) { result.push(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
    }
    for (const auto to : (to_quiet & info.last_rank)) {
      if constexpr (mode::check) { result.push_under_promotions(from, to, piece_type::pawn); }
      if constexpr (mode::noisy) { result.push_queen_promotion(from, to, piece_type::pawn); }
    }
    for (const auto to : (to_noisy & info.last_rank)) {
      if constexpr (mode::check) { result.push_under_promotions(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
      if constexpr (mode::noisy) { result.push_queen_promotion(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_checked_knight(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().knight() & ~info.pinned)) {
    const auto to_mask = knight_attack_tbl.look_up(from);
    const auto to_quiet = info.checker_rays & to_mask;
    const auto to_noisy = info.checkers & to_mask;
    if constexpr (mode::check) {
      for (const auto to : to_quiet) { result.push(from, to, piece_type::knight); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : to_noisy) { result.push(from, to, piece_type::knight, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_checked_rook(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().rook() & ~info.pinned)) {
    const auto to_mask = rook_attack_tbl.look_up(from, info.occ);
    const auto to_quiet = info.checker_rays & to_mask;
    const auto to_noisy = info.checkers & to_mask;
    if constexpr (mode::check) {
      for (const auto to : to_quiet) { result.push(from, to, piece_type::rook); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : to_noisy) { result.push(from, to, piece_type::rook, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_checked_bishop(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().bishop() & ~info.pinned)) {
    const auto to_mask = bishop_attack_tbl.look_up(from, info.occ);
    const auto to_quiet = info.checker_rays & to_mask;
    const auto to_noisy = info.checkers & to_mask;
    if constexpr (mode::check) {
      for (const auto to : to_quiet) { result.push(from, to, piece_type::bishop); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : to_noisy) { result.push(from, to, piece_type::bishop, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_checked_queen(const move_generator_info& info, move_list& result) const noexcept {
  for (const auto from : (man_.us<c>().queen() & ~info.pinned)) {
    const auto to_mask = bishop_attack_tbl.look_up(from, info.occ) | rook_attack_tbl.look_up(from, info.occ);
    const auto to_quiet = info.checker_rays & to_mask;
    const auto to_noisy = info.checkers & to_mask;
    if constexpr (mode::check) {
      for (const auto to : to_quiet) { result.push(from, to, piece_type::queen); }
    }
    if constexpr (mode::noisy) {
      for (const auto to : to_noisy) { result.push(from, to, piece_type::queen, true, man_.them<c>().occ(to)); }
    }
  }
}

template <color c, typename mode>
inline void board::add_king(const move_generator_info& info, move_list& result) const noexcept {
  const square_set to_mask = ~info.king_danger & king_attack_tbl.look_up(man_.us<c>().king().item());
  if (info.checkers.any() ? mode::check : mode::quiet) {
    for (const square to : (to_mask & ~info.occ)) { result.push(man_.us<c>().king().item(), to, piece_type::king); }
  }
  if (mode::noisy) {
    for (const square to : (to_mask & man_.them<c>().all())) {
      result.push(man_.us<c>().king().item(), to, piece_type::king, true, man_.them<c>().occ(to));
    }
  }
}

template <color c>
inline move_generator_info board::get_move_generator_info() const noexcept {
  return move_generator_info{};
}

template <color c, typename mode>
inline move_list board::generate_moves_() const noexcept {
  move_list pseudo{};
  move_list legal{};

  if (!man_.us<c>().king().any()) { return legal; }

  const square_set occ = man_.white.all() | man_.black.all();

  // pawns
  for (const auto from : man_.us<c>().pawn()) {
    const square_set pushes = pawn_push_tbl<c>.look_up(from, occ);
    for (const auto to : pushes) {
      const bool promo = pawn_info<c>::last_rank.is_member(to);
      if (promo) {
        pseudo.push_queen_promotion(from, to, piece_type::pawn);
        pseudo.push_under_promotions(from, to, piece_type::pawn);
      } else {
        pseudo.push(from, to, piece_type::pawn);
      }
    }

    const square_set caps = pawn_attack_tbl<c>.look_up(from) & man_.them<c>().all();
    for (const auto to : caps) {
      const bool promo = pawn_info<c>::last_rank.is_member(to);
      if (promo) {
        pseudo.push_queen_promotion(from, to, piece_type::pawn, true, man_.them<c>().occ(to));
        pseudo.push_under_promotions(from, to, piece_type::pawn, true, man_.them<c>().occ(to));
      } else {
        pseudo.push(from, to, piece_type::pawn, true, man_.them<c>().occ(to));
      }
    }

    if (lat_.them<c>().ep_mask().any()) {
      const square ep_sq = lat_.them<c>().ep_mask().item();
      if (pawn_attack_tbl<c>.look_up(from).is_member(ep_sq)) {
        const square cap_sq = pawn_push_tbl<opponent<c>>.look_up(ep_sq, square_set{}).item();
        pseudo.push(from, ep_sq, piece_type::pawn, true, piece_type::pawn, true, cap_sq);
      }
    }
  }

  // knights
  for (const auto from : man_.us<c>().knight()) {
    const square_set mask = knight_attack_tbl.look_up(from);
    for (const auto to : (mask & ~man_.us<c>().all())) {
      if (man_.them<c>().all().is_member(to)) {
        pseudo.push(from, to, piece_type::knight, true, man_.them<c>().occ(to));
      } else {
        pseudo.push(from, to, piece_type::knight);
      }
    }
  }

  // bishops
  for (const auto from : man_.us<c>().bishop()) {
    const square_set mask = bishop_attack_tbl.look_up(from, occ);
    for (const auto to : (mask & ~man_.us<c>().all())) {
      if (man_.them<c>().all().is_member(to)) {
        pseudo.push(from, to, piece_type::bishop, true, man_.them<c>().occ(to));
      } else {
        pseudo.push(from, to, piece_type::bishop);
      }
    }
  }

  // rooks
  for (const auto from : man_.us<c>().rook()) {
    const square_set mask = rook_attack_tbl.look_up(from, occ);
    for (const auto to : (mask & ~man_.us<c>().all())) {
      if (man_.them<c>().all().is_member(to)) {
        pseudo.push(from, to, piece_type::rook, true, man_.them<c>().occ(to));
      } else {
        pseudo.push(from, to, piece_type::rook);
      }
    }
  }

  // queens
  for (const auto from : man_.us<c>().queen()) {
    const square_set mask = bishop_attack_tbl.look_up(from, occ) | rook_attack_tbl.look_up(from, occ);
    for (const auto to : (mask & ~man_.us<c>().all())) {
      if (man_.them<c>().all().is_member(to)) {
        pseudo.push(from, to, piece_type::queen, true, man_.them<c>().occ(to));
      } else {
        pseudo.push(from, to, piece_type::queen);
      }
    }
  }

  // king (only quiet moves; king captures are illegal in atomic)
  for (const auto to : (king_attack_tbl.look_up(man_.us<c>().king().item()) & ~occ)) {
    pseudo.push(man_.us<c>().king().item(), to, piece_type::king);
  }

  // castling
  if (lat_.us<c>().oo() && (castle_info<c>.oo_mask & occ).count() == 0) {
    pseudo.push(castle_info<c>.start_king, castle_info<c>.oo_rook, piece_type::king);
  }
  if (lat_.us<c>().ooo() && (castle_info<c>.ooo_occ_mask & occ).count() == 0) {
    pseudo.push(castle_info<c>.start_king, castle_info<c>.ooo_rook, piece_type::king);
  }

  for (const auto& mv : pseudo) {
    if (is_legal_<c, mode>(mv)) { legal.push(mv); }
  }

  return legal;
}

template <typename mode>
move_list board::generate_moves() const noexcept {
  return turn() ? generate_moves_<color::white, mode>() : generate_moves_<color::black, mode>();
}

template <color c, typename mode>
inline bool board::is_legal_(const move& mv) const noexcept {
  if (!man_.us<c>().king().any()) { return false; }

  // Handle castling separately (destination is occupied by our rook)
  if (mv.is_castle_oo<c>() || mv.is_castle_ooo<c>()) {
    const bool short_castle = mv.is_castle_oo<c>();

    // basic consistency: no capture/ep/promotion and rights/pieces present
    if (mv.is_capture() || mv.is_enpassant() || mv.is_promotion()) { return false; }
    if (mv.from() != castle_info<c>.start_king) { return false; }
    if (short_castle && mv.to() != castle_info<c>.oo_rook) { return false; }
    if (!short_castle && mv.to() != castle_info<c>.ooo_rook) { return false; }
    if (!man_.us<c>().king().is_member(castle_info<c>.start_king)) { return false; }
    if (short_castle && !man_.us<c>().rook().is_member(castle_info<c>.oo_rook)) { return false; }
    if (!short_castle && !man_.us<c>().rook().is_member(castle_info<c>.ooo_rook)) { return false; }
    if (short_castle && !lat_.us<c>().oo()) { return false; }
    if (!short_castle && !lat_.us<c>().ooo()) { return false; }

    const square_set occ = man_.white.all() | man_.black.all();
    if (short_castle) {
      if ((castle_info<c>.oo_mask & occ).any()) { return false; }
    } else {
      if ((castle_info<c>.ooo_occ_mask & occ).any()) { return false; }
    }

    // cannot castle out of or through direct check
    if (std::get<0>(checkers<c>(occ)).any()) { return false; }
    const square_set danger_path = short_castle ? castle_info<c>.oo_mask : castle_info<c>.ooo_danger_mask;
    for (const auto sq : danger_path) {
      if (attack_to<opponent<c>>(*this, sq, occ).any()) { return false; }
    }

    // Final position must also be free of direct check (and allow double-king blast logic)
    const board next = forward_<c>(mv);
    const bool us_dead = !next.man_.us<c>().king().any();
    const bool them_dead = !next.man_.them<c>().king().any();
    const square_set occ_after = next.man_.white.all() | next.man_.black.all();
    if (us_dead && !them_dead) { return false; }
    if (!them_dead && !us_dead) {
      const auto [checks, _] = next.checkers<c>(occ_after);
      if (checks.any()) { return false; }
    }
    return true;
  }

  if (!man_.us<c>().all().is_member(mv.from())) { return false; }
  if (man_.us<c>().all().is_member(mv.to())) { return false; }
  if (mv.piece() != man_.us<c>().occ(mv.from())) { return false; }

  const bool to_has_enemy = man_.them<c>().all().is_member(mv.to());

  if (mv.is_capture() != (to_has_enemy || mv.is_enpassant())) { return false; }
  if (!mv.is_capture() && mv.captured() != static_cast<piece_type>(0)) { return false; }
  if (mv.is_capture() && !mv.is_enpassant()) {
    if (!to_has_enemy) { return false; }
    if (mv.captured() != man_.them<c>().occ(mv.to())) { return false; }
  }

  if (!mv.is_enpassant() && mv.enpassant_sq() != square::from_index(0)) { return false; }
  if (mv.is_enpassant()) {
    if (!lat_.them<c>().ep_mask().any() || !lat_.them<c>().ep_mask().is_member(mv.to())) { return false; }
    const square cap_sq = pawn_push_tbl<opponent<c>>.look_up(mv.to(), square_set{}).item();
    if (mv.enpassant_sq() != cap_sq) { return false; }
    if (!man_.them<c>().pawn().is_member(cap_sq)) { return false; }
  }

  const square_set occ = man_.white.all() | man_.black.all();

  const bool legal_from_to = [&] {
    switch (mv.piece()) {
      case piece_type::pawn: {
        if (mv.is_capture()) { return pawn_attack_tbl<c>.look_up(mv.from()).is_member(mv.to()); }
        return pawn_push_tbl<c>.look_up(mv.from(), occ).is_member(mv.to());
      }
      case piece_type::knight: return knight_attack_tbl.look_up(mv.from()).is_member(mv.to());
      case piece_type::bishop: return bishop_attack_tbl.look_up(mv.from(), occ).is_member(mv.to());
      case piece_type::rook: return rook_attack_tbl.look_up(mv.from(), occ).is_member(mv.to());
      case piece_type::queen: return (bishop_attack_tbl.look_up(mv.from(), occ) | rook_attack_tbl.look_up(mv.from(), occ)).is_member(mv.to());
      case piece_type::king: {
        if (mv.is_capture()) { return false; }
        if (mv.is_castle_oo<c>()) {
          return lat_.us<c>().oo() && (castle_info<c>.oo_mask & occ).count() == 0 && man_.us<c>().king().is_member(castle_info<c>.start_king) &&
                 man_.us<c>().rook().is_member(castle_info<c>.oo_rook);
        }
        if (mv.is_castle_ooo<c>()) {
          return lat_.us<c>().ooo() && (castle_info<c>.ooo_occ_mask & occ).count() == 0 &&
                 man_.us<c>().king().is_member(castle_info<c>.start_king) && man_.us<c>().rook().is_member(castle_info<c>.ooo_rook);
        }
        return king_attack_tbl.look_up(mv.from()).is_member(mv.to());
      }
      default: return false;
    }
  }();

  if (!legal_from_to) { return false; }

  if (mv.is_promotion()) {
    if (mv.piece() != piece_type::pawn) { return false; }
    if (!pawn_info<c>::last_rank.is_member(mv.to())) { return false; }
    if (mv.promotion() < piece_type::knight || mv.promotion() > piece_type::queen) { return false; }
  } else if (mv.piece() == piece_type::pawn && pawn_info<c>::last_rank.is_member(mv.to())) {
    return false;
  }

  const bool is_noisy = mv.is_noisy();
  if (!mode::noisy && is_noisy) { return false; }
  if (!mode::quiet && !is_noisy) { return false; }

  // Captures that would explode our own king are illegal, even if the enemy king also dies
  if (mv.is_capture() || mv.is_enpassant()) {
    // For en passant, explosion is centered on the captured pawn's square
    const square explosion_center = mv.is_enpassant()
      ? pawn_push_tbl<opponent<c>>.look_up(mv.to(), square_set{}).item()
      : mv.to();

    const square_set blast = explosion_mask(explosion_center);
    if ((blast & man_.us<c>().king()).any()) { return false; }
  }

  const board next = forward_<c>(mv);
  const bool us_dead = !next.man_.us<c>().king().any();
  const bool them_dead = !next.man_.them<c>().king().any();
  const square_set occ_after = next.man_.white.all() | next.man_.black.all();

  // Illegal if we die and they survive; if both kings die, allow (win)
  if (us_dead && !them_dead) { return false; }
  if (!them_dead && !us_dead) {
    const bool kings_touch = [&] {
      const square_set our_k = next.man_.us<c>().king();
      const square_set their_k = next.man_.them<c>().king();
      if (!our_k.any() || !their_k.any()) { return false; }
      return king_attack_tbl.look_up(our_k.item()).is_member(their_k.item());
    }();

    if (!kings_touch) {
      const auto [checks, _] = next.checkers<c>(occ_after);
      if (checks.any()) { return false; }
    }
  }

  return true;
}

template <typename mode>
bool board::is_legal(const move& mv) const noexcept {
  return turn() ? is_legal_<color::white, mode>(mv) : is_legal_<color::black, mode>(mv);
}

template <color c>
[[nodiscard]] inline bool board::upcoming_cycle_exists_(const std::size_t& height, const board_history& history) const noexcept {
  const std::size_t size = history.future_size(height);
  const std::size_t limit = std::min(size, lat_.half_clock);

  const sided_zobrist_hash hash = sided_hash();
  const zobrist::hash_type us_hash = hash.us<c>();
  const zobrist::hash_type them_hash = hash.them<c>();

  for (std::size_t reverse_idx = 3; reverse_idx <= limit; reverse_idx += 2) {
    const std::size_t idx = size - reverse_idx;
    const zobrist::hash_type delta = us_hash ^ history.at(idx).us<c>();

    if (delta == cuckoo_hash_table::value_type::initial_hash) { continue; }
    if (them_hash != history.at(idx).them<c>()) { continue; }

    const std::optional<cuckoo_hash_table_entry> entry = cuckoo_hash_table::instance.look_up(delta);

    if (entry.has_value()) {
      const square_set candidates = square_set::of(entry->one(), entry->two());
      const square_set available = man_.us<c>().get_plane(entry->piece());

      if (!(available & candidates).any()) { return false; }

      const square_set occ = man_.white.all() | man_.black.all();
      const square_set ray = ray_between_tbl.look_up(entry->one(), entry->two());

      if (!(occ & ray).any()) { return true; }
    }
  }

  return false;
}

[[nodiscard]] bool board::upcoming_cycle_exists(const std::size_t& height, const board_history& history) const noexcept {
  return turn() ? upcoming_cycle_exists_<color::white>(height, history) : upcoming_cycle_exists_<color::black>(height, history);
}

template <color c>
inline bool board::is_check_() const noexcept {
  if (!man_.us<c>().king().any()) { return true; }
  // kings touching means no direct check applies
  if (man_.us<c>().king().any() && man_.them<c>().king().any()) {
    if (king_attack_tbl.look_up(man_.us<c>().king().item()).is_member(man_.them<c>().king().item())) { return false; }
  }
  return king_capturable_in_position<opponent<c>>(*this);
}

bool board::is_check() const noexcept { return turn() ? is_check_<color::white>() : is_check_<color::black>(); }

template <color c, typename T>
inline bool board::see_ge_(const move& mv, const T& threshold) const noexcept {
  using score_t = std::int32_t;
  constexpr score_t score_mate = 1'000'000;

  static constexpr score_t values[] = {
      100,   // pawn
      450,   // knight
      450,   // bishop
      650,   // rook
      1250,  // queen
      0      // king
  };

  auto value = [&](const piece_type pt) -> score_t { return values[static_cast<int>(pt)]; };

  if (mv.is_null()) { return true; }

  const color us = c;
  const score_t thr = static_cast<score_t>(threshold);

  const bool is_capture = mv.is_capture() || mv.is_enpassant();
  const bool is_castle = mv.is_castle_oo<c>() || mv.is_castle_ooo<c>();

  // CAPTURES: In atomic chess, captures cause explosions
  if (is_capture && !is_castle) {
    score_t score{0};
    square explosion_center = mv.to();
    square_set pieces_exploded = square_set::of(mv.to());  // Capturing piece always explodes

    if (mv.is_enpassant()) {
      // En passant: explosion is centered on the captured pawn's square
      explosion_center = pawn_push_tbl<opponent<c>>.look_up(mv.to(), square_set{}).item();
      // Gain the captured pawn
      score += value(piece_type::pawn);
    } else {
      // Normal capture: gain the captured piece (it's at mv.to())
      score += value(mv.captured());
    }

    // Calculate explosion area (center + 8 adjacent squares), excluding pawns
    const square_set pawns_all = man_.white.pawn() | man_.black.pawn();
    const square_set explosion_area = explosion_mask(explosion_center) & ~pawns_all;

    // All non-pawn pieces in explosion area are removed, plus the capturing piece
    const square_set boom = explosion_area | pieces_exploded;

    // Check if kings die in the explosion
    if ((boom & man_.us<us>().king()).any()) { return -score_mate >= thr; }
    if ((boom & man_.them<us>().king()).any()) { return score_mate >= thr; }

    // Calculate material gain/loss from explosion
    square_set boom_us = boom & man_.us<us>().all();
    square_set boom_them = boom & man_.them<us>().all();

    for (const auto sq : boom_us) { score -= value(piece_at_unchecked(*this, sq)); }
    for (const auto sq : boom_them) { score += value(piece_at_unchecked(*this, sq)); }

    return score >= thr;
  }

  // QUIET MOVES and CASTLING: No explosions in atomic chess for non-captures
  // For quiet moves, SEE is trivial - we just check if the move loses material to a recapture

  // Castling is always safe in terms of material (no piece is lost)
  if (is_castle) { return 0 >= thr; }

  // For quiet moves, the piece just moves from 'from' to 'to'
  // The only risk is if the opponent can capture it on 'to'

  // Calculate position after the quiet move
  const board next = forward_<c>(mv);
  const square_set occ_after = next.man_.white.all() | next.man_.black.all();

  // Check if opponent can capture on mv.to()
  square_set attackers = attack_to<opponent<c>>(next, mv.to(), occ_after);

  if (!attackers.any()) {
    // No attacker, move is safe
    return 0 >= thr;
  }

  // Find the least valuable attacker
  score_t min_attacker_value = score_mate;
  for (const auto sq : attackers) {
    const piece_type pt = piece_at_unchecked(next, sq);
    if (pt == piece_type::king) { continue; }  // Kings can't capture in atomic
    const score_t v = value(pt);
    if (v < min_attacker_value) { min_attacker_value = v; }
  }

  if (min_attacker_value == score_mate) {
    // Only king can attack, which can't capture in atomic
    return 0 >= thr;
  }

  // If opponent captures our piece on mv.to(), there will be an explosion
  // The opponent loses their attacking piece, we lose our moved piece
  // Plus any pieces in the explosion radius (excluding pawns)

  const piece_type our_moved_piece = mv.is_promotion<c>() ? mv.promotion() : mv.piece();
  score_t result = -value(our_moved_piece);  // We lose our piece
  result += min_attacker_value;  // Opponent loses their attacker

  // Calculate explosion if opponent recaptures
  const square_set pawns_all_after = next.man_.white.pawn() | next.man_.black.pawn();
  const square_set explosion_area = explosion_mask(mv.to()) & ~pawns_all_after;
  const square_set boom = explosion_area | square_set::of(mv.to());  // Attacker also explodes

  // Check if recapture would kill our king
  if ((boom & next.man_.us<c>().king()).any()) {
    // Opponent can kill our king by recapturing
    return -score_mate >= thr;
  }

  // Check if recapture would kill opponent's king
  if ((boom & next.man_.them<c>().king()).any()) {
    // Opponent can't recapture because it would kill their own king
    return 0 >= thr;
  }

  // Add/subtract pieces lost in the explosion
  square_set boom_us = boom & next.man_.us<c>().all();
  square_set boom_them = boom & next.man_.them<c>().all();

  for (const auto sq : boom_us) { result -= value(piece_at_unchecked(next, sq)); }
  for (const auto sq : boom_them) { result += value(piece_at_unchecked(next, sq)); }

  return result >= thr;
}

template <typename T>
bool board::see_ge(const move& mv, const T& threshold) const noexcept {
  return turn() ? see_ge_<color::white, T>(mv, threshold) : see_ge_<color::black, T>(mv, threshold);
}

template <typename T>
bool board::see_gt(const move& mv, const T& threshold) const noexcept {
  return see_ge(mv, threshold + 1);
}

template <typename T>
T board::phase() const noexcept {
  static_assert(std::is_floating_point_v<T>);
  constexpr T start_pos_value = static_cast<T>(24);

  T value{};
  over_types([&](const piece_type& pt) { value += phase_value<T>(pt) * (man_.white.get_plane(pt) | man_.black.get_plane(pt)).count(); });
  return std::min(value, start_pos_value) / start_pos_value;
}

bool board::has_non_pawn_material() const noexcept {
  return man_.us(turn()).knight().any() || man_.us(turn()).bishop().any() || man_.us(turn()).rook().any() || man_.us(turn()).queen().any();
}

template <color c>
inline bool board::is_passed_push_(const move& mv) const noexcept {
  return ((mv.piece() == piece_type::pawn && !mv.is_capture()) && !(man_.them<c>().pawn() & passer_tbl<c>.mask(mv.to())).any());
}

bool board::is_passed_push(const move& mv) const noexcept { return turn() ? is_passed_push_<color::white>(mv) : is_passed_push_<color::black>(mv); }

template <color c>
std::size_t board::side_num_pieces() const noexcept {
  return man_.us<c>().all().count();
}

std::size_t board::num_pieces() const noexcept { return side_num_pieces<color::white>() + side_num_pieces<color::black>(); }

bool board::is_trivially_drawn() const noexcept {
  return false;
}

template <color c>
board board::forward_(const move& mv) const noexcept {
  board copy = *this;
  if (mv.is_null()) {
    ++copy.lat_.ply_count;
    ++copy.lat_.half_clock;
    return copy;
  }

  const bool is_castle_q = mv.is_castle_ooo<c>();
  const bool is_castle_k = mv.is_castle_oo<c>();

  piece_type placed_piece = mv.piece();

  copy.man_.us<c>().remove_piece(mv.piece(), mv.from());

  if (is_castle_q) {
    copy.lat_.us<c>().set_ooo(false).set_oo(false);
    copy.man_.us<c>().remove_piece(piece_type::rook, castle_info<c>.ooo_rook);
    copy.man_.us<c>().add_piece(piece_type::king, castle_info<c>.after_ooo_king);
    copy.man_.us<c>().add_piece(piece_type::rook, castle_info<c>.after_ooo_rook);
  } else if (is_castle_k) {
    copy.lat_.us<c>().set_ooo(false).set_oo(false);
    copy.man_.us<c>().remove_piece(piece_type::rook, castle_info<c>.oo_rook);
    copy.man_.us<c>().add_piece(piece_type::king, castle_info<c>.after_oo_king);
    copy.man_.us<c>().add_piece(piece_type::rook, castle_info<c>.after_oo_rook);
  } else {
    if (mv.is_promotion<c>()) { placed_piece = mv.promotion(); }
    copy.man_.us<c>().add_piece(placed_piece, mv.to());
  }

  if (mv.is_pawn_double<c>()) {
    const square ep = pawn_push_tbl<opponent<c>>.look_up(mv.to(), square_set{}).item();
    if ((man_.them<c>().pawn() & pawn_attack_tbl<c>.look_up(ep)).any()) { copy.lat_.us<c>().set_ep_mask(ep); }
  }

  if (mv.from() == castle_info<c>.start_king) { copy.lat_.us<c>().set_ooo(false).set_oo(false); }
  if (mv.from() == castle_info<c>.oo_rook) { copy.lat_.us<c>().set_oo(false); }
  if (mv.from() == castle_info<c>.ooo_rook) { copy.lat_.us<c>().set_ooo(false); }

  if (mv.is_capture() || mv.is_enpassant()) {
    // For en passant, explosion is centred on the captured pawn's square, not mv.to()
    square explosion_center = mv.to();

    if (mv.is_enpassant()) {
      explosion_center = mv.enpassant_sq();
      copy.man_.them<c>().remove_piece(piece_type::pawn, mv.enpassant_sq());
    } else {
      copy.man_.them<c>().remove_piece(mv.captured(), mv.to());
    }

    // Remove the capturing piece from its destination
    copy.man_.us<c>().remove_piece(placed_piece, mv.to());

    // Calculate explosion area: center + 8 adjacent squares (but NOT pawns)
    const square_set blast = explosion_mask(explosion_center);

    // Explicitly copy piece sets before iterating to avoid any iterator issues
    const square_set white_knights_in_blast = blast & copy.man_.white.knight();
    const square_set white_bishops_in_blast = blast & copy.man_.white.bishop();
    const square_set white_rooks_in_blast = blast & copy.man_.white.rook();
    const square_set white_queens_in_blast = blast & copy.man_.white.queen();
    const square_set white_kings_in_blast = blast & copy.man_.white.king();
    const square_set black_knights_in_blast = blast & copy.man_.black.knight();
    const square_set black_bishops_in_blast = blast & copy.man_.black.bishop();
    const square_set black_rooks_in_blast = blast & copy.man_.black.rook();
    const square_set black_queens_in_blast = blast & copy.man_.black.queen();
    const square_set black_kings_in_blast = blast & copy.man_.black.king();

    // Remove all non-pawn pieces in blast area
    for (const auto sq : white_knights_in_blast) { copy.man_.white.remove_piece(piece_type::knight, sq); }
    for (const auto sq : white_bishops_in_blast) { copy.man_.white.remove_piece(piece_type::bishop, sq); }
    for (const auto sq : white_rooks_in_blast) { copy.man_.white.remove_piece(piece_type::rook, sq); }
    for (const auto sq : white_queens_in_blast) { copy.man_.white.remove_piece(piece_type::queen, sq); }
    for (const auto sq : white_kings_in_blast) { copy.man_.white.remove_piece(piece_type::king, sq); }
    for (const auto sq : black_knights_in_blast) { copy.man_.black.remove_piece(piece_type::knight, sq); }
    for (const auto sq : black_bishops_in_blast) { copy.man_.black.remove_piece(piece_type::bishop, sq); }
    for (const auto sq : black_rooks_in_blast) { copy.man_.black.remove_piece(piece_type::rook, sq); }
    for (const auto sq : black_queens_in_blast) { copy.man_.black.remove_piece(piece_type::queen, sq); }
    for (const auto sq : black_kings_in_blast) { copy.man_.black.remove_piece(piece_type::king, sq); }
  }

  if (copy.lat_.white.oo() && !copy.man_.white.rook().is_member(castle_info<color::white>.oo_rook)) { copy.lat_.white.set_oo(false); }
  if (copy.lat_.white.ooo() && !copy.man_.white.rook().is_member(castle_info<color::white>.ooo_rook)) { copy.lat_.white.set_ooo(false); }
  if (copy.lat_.black.oo() && !copy.man_.black.rook().is_member(castle_info<color::black>.oo_rook)) { copy.lat_.black.set_oo(false); }
  if (copy.lat_.black.ooo() && !copy.man_.black.rook().is_member(castle_info<color::black>.ooo_rook)) { copy.lat_.black.set_ooo(false); }

  if (mv.to() == castle_info<opponent<c>>.oo_rook) { copy.lat_.them<c>().set_oo(false); }
  if (mv.to() == castle_info<opponent<c>>.ooo_rook) { copy.lat_.them<c>().set_ooo(false); }

  copy.lat_.them<c>().clear_ep_mask();
  ++copy.lat_.ply_count;
  ++copy.lat_.half_clock;
  if (mv.is_capture() || mv.piece() == piece_type::pawn) { copy.lat_.half_clock = 0; }
  return copy;
}

board board::forward(const move& mv) const noexcept { return turn() ? forward_<color::white>(mv) : forward_<color::black>(mv); }

board board::mirrored() const noexcept {
  board mirror{};
  // manifest
  over_types([&mirror, this](const piece_type& pt) {
    for (const auto sq : man_.white.get_plane(pt).mirrored()) { mirror.man_.black.add_piece(pt, sq); }
    for (const auto sq : man_.black.get_plane(pt).mirrored()) { mirror.man_.white.add_piece(pt, sq); }
  });
  // latent
  mirror.lat_.white.set_ooo(lat_.black.ooo());
  mirror.lat_.black.set_ooo(lat_.white.ooo());
  mirror.lat_.white.set_oo(lat_.black.oo());
  mirror.lat_.black.set_oo(lat_.white.oo());
  if (lat_.black.ep_mask().any()) { mirror.lat_.white.set_ep_mask(lat_.black.ep_mask().mirrored().item()); }
  if (lat_.white.ep_mask().any()) { mirror.lat_.black.set_ep_mask(lat_.white.ep_mask().mirrored().item()); }
  mirror.lat_.ply_count = lat_.ply_count ^ static_cast<std::size_t>(1);
  mirror.lat_.half_clock = lat_.half_clock;

  return mirror;
}

std::tuple<board_history, board> board::after_uci_moves(const std::string& moves) const noexcept {
  board_history history{};
  auto bd = *this;

  std::istringstream move_stream(moves);
  std::string move_name;

  while (move_stream >> move_name) {
    const move_list list = bd.generate_moves<>();
    const auto it = std::find_if(list.begin(), list.end(), [=](const move& mv) { return mv.name(bd.turn()) == move_name; });
    assert((it != list.end()));
    history.push(bd.sided_hash());
    bd = bd.forward(*it);
  }
  return std::tuple(history, bd);
}

std::string board::fen() const noexcept {
  std::string fen{};
  constexpr std::size_t num_ranks = 8;
  for (std::size_t i{0}; i < num_ranks; ++i) {
    std::size_t j{0};
    over_rank(i, [&, this](const tbl_square& at_r) {
      const tbl_square at = at_r.rotated();
      if (man_.white.all().occ(at.index())) {
        const char letter = piece_letter(color::white, man_.white.occ(at));
        if (j != 0) { fen.append(std::to_string(j)); }
        fen.push_back(letter);
        j = 0;
      } else if (man_.black.all().occ(at.index())) {
        const char letter = piece_letter(color::black, man_.black.occ(at));
        if (j != 0) { fen.append(std::to_string(j)); }
        fen.push_back(letter);
        j = 0;
      } else {
        ++j;
      }
    });
    if (j != 0) { fen.append(std::to_string(j)); }
    if (i != (num_ranks - 1)) { fen.push_back('/'); }
  }
  fen.push_back(' ');
  fen.push_back(turn() ? 'w' : 'b');
  fen.push_back(' ');
  std::string castle_rights{};
  if (lat_.white.oo()) { castle_rights.push_back('K'); }
  if (lat_.white.ooo()) { castle_rights.push_back('Q'); }
  if (lat_.black.oo()) { castle_rights.push_back('k'); }
  if (lat_.black.ooo()) { castle_rights.push_back('q'); }
  fen.append(castle_rights.empty() ? "-" : castle_rights);
  fen.push_back(' ');
  fen.append(lat_.them(turn()).ep_mask().any() ? lat_.them(turn()).ep_mask().item().name() : "-");
  fen.push_back(' ');
  fen.append(std::to_string(lat_.half_clock));
  fen.push_back(' ');
  fen.append(std::to_string(1 + (lat_.ply_count / 2)));
  return fen;
}

board board::start_pos() noexcept { return parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

board board::parse_fen(const std::string& fen) noexcept {
  auto fen_pos = board();
  std::stringstream ss(fen);

  std::string body;
  ss >> body;
  std::string side;
  ss >> side;
  std::string castle;
  ss >> castle;
  std::string ep_sq;
  ss >> ep_sq;
  std::string half_clock;
  ss >> half_clock;
  std::string move_count;
  ss >> move_count;
  {
    std::stringstream body_s(body);
    std::string rank;
    for (int rank_idx{0}; std::getline(body_s, rank, '/'); ++rank_idx) {
      int file_idx{0};
      for (const char c : rank) {
        if (std::isdigit(c)) {
          file_idx += static_cast<int>(c - '0');
        } else {
          const color side = color_from(c);
          const piece_type type = type_from(c);
          const tbl_square sq = tbl_square{file_idx, rank_idx}.rotated();
          fen_pos.man_.us(side).add_piece(type, sq);
          ++file_idx;
        }
      }
    }
  }
  fen_pos.lat_.white.set_oo(castle.find('K') != std::string::npos);
  fen_pos.lat_.white.set_ooo(castle.find('Q') != std::string::npos);
  fen_pos.lat_.black.set_oo(castle.find('k') != std::string::npos);
  fen_pos.lat_.black.set_ooo(castle.find('q') != std::string::npos);
  fen_pos.lat_.half_clock = std::stol(half_clock);
  if (ep_sq != "-") { fen_pos.lat_.them(side == "w").set_ep_mask(tbl_square::from_name(ep_sq)); }
  fen_pos.lat_.ply_count = static_cast<std::size_t>(2 * (std::stol(move_count) - 1) + static_cast<std::size_t>(side != "w"));
  return fen_pos;
}

std::ostream& operator<<(std::ostream& ostr, const board& bd) noexcept {
  ostr << std::boolalpha;
  ostr << "board(hash=" << bd.hash();
  ostr << ", half_clock=" << bd.lat_.half_clock;
  ostr << ", ply_count=" << bd.lat_.ply_count;
  ostr << ", white.oo_=" << bd.lat_.white.oo();
  ostr << ", white.ooo_=" << bd.lat_.white.ooo();
  ostr << ", black.oo_=" << bd.lat_.black.oo();
  ostr << ", black.ooo_=" << bd.lat_.black.ooo();
  ostr << ",\nwhite.ep_mask=" << bd.lat_.white.ep_mask();
  ostr << ",\nblack.ep_mask=" << bd.lat_.black.ep_mask();
  ostr << "white.occ_table={";
  over_all([&ostr, bd](const tbl_square& sq) { ostr << piece_name(bd.man_.white.occ(sq)) << ", "; });
  ostr << "},\nblack.occ_table={";
  over_all([&ostr, bd](const tbl_square& sq) { ostr << piece_name(bd.man_.black.occ(sq)) << ", "; });
  ostr << "}\n";
  over_types([&ostr, bd](const piece_type& pt) { ostr << "white." << piece_name(pt) << "=" << bd.man_.white.get_plane(pt) << ",\n"; });
  ostr << "white.all=" << bd.man_.white.all() << ",\n";
  over_types([&ostr, bd](const piece_type& pt) { ostr << "black." << piece_name(pt) << "=" << bd.man_.black.get_plane(pt) << ",\n"; });
  ostr << "black.all=" << bd.man_.black.all() << ")";
  return ostr << std::noboolalpha;
}

}  // namespace chess

template chess::move_list chess::board::generate_moves<chess::generation_mode::all>() const noexcept;
template chess::move_list chess::board::generate_moves<chess::generation_mode::noisy_and_check>() const noexcept;
template chess::move_list chess::board::generate_moves<chess::generation_mode::quiet_and_check>() const noexcept;

template bool chess::board::is_legal<chess::generation_mode::all>(const chess::move&) const noexcept;
template bool chess::board::is_legal<chess::generation_mode::noisy_and_check>(const chess::move&) const noexcept;

template bool chess::board::see_ge<std::int32_t>(const chess::move&, const std::int32_t&) const noexcept;
template bool chess::board::see_gt<std::int32_t>(const chess::move&, const std::int32_t&) const noexcept;

template float chess::board::phase<float>() const noexcept;
template double chess::board::phase<double>() const noexcept;
