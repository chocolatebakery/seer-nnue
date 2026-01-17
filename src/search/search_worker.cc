/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

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

#include <search/move_orderer.h>
#include <search/search_worker.h>
#include <search/syzygy.h>
#include <zobrist/zobrist_hasher.h>

namespace {
// Convert Bullet NNUE centipawn output to Seer's internal logit scale.
constexpr search::score_type nnue_scale_numerator = 1024;
constexpr search::score_type nnue_scale_denominator = 288;

inline search::score_type scale_nnue_score(const eval::i32 raw) {
  return static_cast<search::score_type>(raw) * nnue_scale_numerator / nnue_scale_denominator;
}
}  // namespace

namespace search {

template <bool is_pv, bool use_tt>
inline evaluate_info search_worker::evaluate(
    const stack_view& ss,
    eval::NnueState& nnue_state,
    const chess::board& bd,
    const std::optional<transposition_table_entry>& maybe) noexcept {
  const bool is_check = bd.is_check() || bd.in_atomic_blast_check();

  const eval_cache_entry entry = [&] {
    constexpr zobrist::hash_type default_hash = zobrist::hash_type{};

    if (is_check) { return eval_cache_entry::make(default_hash, default_hash, ss.loss_score()); }
    if (const auto maybe_eval = internal.cache.find(bd.hash()); !is_pv && maybe_eval.has_value()) { return maybe_eval.value(); }

    const zobrist::hash_type hash = bd.hash();
    const chess::color stm = bd.turn() ? chess::color::white : chess::color::black;
    const score_type eval = scale_nnue_score(nnue_state.evaluate(bd, stm));

    constexpr std::size_t feature_hash_dim = 256;
    static_assert(eval::Layer1Size >= feature_hash_dim);

    const auto outputs = nnue_state.outputs(stm);
    const auto eval_feature_hash =
        zobrist::zobrist_hasher<zobrist::quarter_hash_type, feature_hash_dim>.compute_hash(
            [&outputs](const std::size_t& i) { return outputs[i] > 0; });

    return eval_cache_entry::make(hash, eval_feature_hash, eval);
  }();

  const auto pawn_feature_hash = zobrist::lower_quarter(bd.pawn_hash());
  const auto eval_feature_hash = entry.eval_feature_hash();

  const auto feature_hash = composite_feature_hash_of(pawn_feature_hash, eval_feature_hash);
  score_type static_value = entry.eval();

  if (!is_check) {
    internal.cache.insert(bd.hash(), entry);
    static_value += internal.correction.us(bd.turn()).correction_for(feature_hash);
  }

  score_type value = static_value;

  if (use_tt && maybe.has_value()) {
    if (maybe->bound() == bound_type::upper && static_value > maybe->score()) { value = maybe->score(); }
    if (maybe->bound() == bound_type::lower && static_value < maybe->score()) { value = maybe->score(); }
  }

  return evaluate_info{feature_hash, static_value, value};
}

template <bool is_pv, bool use_tt>
score_type search_worker::q_search(
    const stack_view& ss,
    eval::NnueState& nnue_state,
    const chess::board& bd,
    score_type alpha,
    const score_type& beta,
    const depth_type& elevation) noexcept {
  // callback on entering search function
  const bool should_update = internal.keep_going() && internal.one_of<nodes_per_update>();
  if (should_update) { external.on_update(*this); }

  ++internal.nodes;
  if (!bd.man_.us(bd.turn()).king().any()) { return ss.loss_score(); }
  if (!bd.man_.them(bd.turn()).king().any()) { return ss.win_score(); }
  const bool is_check = bd.is_check();
  const bool atomic_check = bd.in_atomic_blast_check();
  const bool is_check_any = is_check || atomic_check;

  if (bd.is_trivially_drawn()) { return draw_score; }
  if (ss.upcoming_cycle_exists(bd)) {
    if (draw_score >= beta) { return draw_score; }
    alpha = std::max(draw_score, alpha);
  }

  const std::optional<transposition_table_entry> maybe = external.tt->find(bd.hash());
  if (maybe.has_value()) {
    const transposition_table_entry entry = maybe.value();
    const bool is_cutoff = (entry.bound() == bound_type::lower && entry.score() >= beta) || (entry.bound() == bound_type::exact) ||
                           (entry.bound() == bound_type::upper && entry.score() <= alpha);
    if (use_tt && is_cutoff) { return entry.score(); }
  }

  const auto [feature_hash, static_value, value] = evaluate<is_pv, use_tt>(ss, nnue_state, bd, maybe);

  if (!is_check_any && value >= beta) { return value; }
  if (ss.reached_max_height()) { return value; }

  move_orderer<chess::generation_mode::noisy_and_check> orderer(move_orderer_data(&bd, &internal.hh.us(bd.turn())));
  if (maybe.has_value()) { orderer.set_first(maybe->best_move()); }

  alpha = std::max(alpha, value);
  score_type best_score = value;
  chess::move best_move = chess::move::null();

  ss.set_hash(bd.sided_hash()).set_eval(static_value);
  int legal_count{0};
  for (const auto& [idx, mv] : orderer) {
    ++legal_count;
    if (!internal.keep_going()) { break; }

    const bool blast_mate = bd.is_atomic_king_blast_capture(mv);

    if (!is_check_any && !bd.see_ge(mv, 0) && !blast_mate) { break; }

    const bool delta_prune =
        !is_pv && !is_check_any && !blast_mate && !bd.see_gt(mv, 0) && ((value + external.constants->delta_margin()) < alpha);
    if (delta_prune) { break; }

    const bool good_capture_prune = !is_pv && !is_check_any && !blast_mate && !maybe.has_value() &&
                                    bd.see_ge(mv, external.constants->good_capture_prune_see_margin()) &&
                                    value + external.constants->good_capture_prune_score_margin() > beta;
    if (good_capture_prune) { return beta; }

    ss.set_played(mv);

    if (blast_mate) { return ss.win_score(); }

    const chess::board bd_ = bd.forward(mv);
    external.tt->prefetch(bd_.hash());
    internal.cache.prefetch(bd_.hash());
    const auto updates = eval::buildUpdates(bd, bd_);
    nnue_state.update<true>(updates, bd_);
    const score_type score = -q_search<is_pv, use_tt>(ss.next(), nnue_state, bd_, -beta, -alpha, elevation + 1);
    nnue_state.pop();

    if (score > best_score) {
      best_score = score;
      best_move = mv;
      if (score > alpha) {
        if (score < beta) { alpha = score; }
        if constexpr (is_pv) { ss.prepend_to_pv(mv); }
      }
    }

    if (best_score >= beta) { break; }
  }

  // Promotion quiescence: include quiet under-promotions/pushes that are otherwise skipped by noisy move gen
  if (!is_check_any && best_score < beta && elevation == 0 && best_score + 100 >= alpha && internal.keep_going()) {
    constexpr int promo_limit = 6;
    int explored_promos = 0;

    const chess::move_list quiets = bd.generate_moves<chess::generation_mode::quiet_and_check>();
    for (const auto& mv : quiets) {
      if (!mv.is_promotion() || mv.is_noisy()) { continue; }
      if (explored_promos >= promo_limit) { break; }

      ++explored_promos;
      const chess::board bd_promo = bd.forward(mv);

      ss.set_played(mv);
      const auto updates = eval::buildUpdates(bd, bd_promo);
      nnue_state.update<true>(updates, bd_promo);
      const score_type score = -q_search<is_pv, use_tt>(ss.next(), nnue_state, bd_promo, -beta, -alpha, elevation + 1);
      nnue_state.pop();

      if (score > best_score) {
        best_score = score;
        best_move = mv;
        if (score > alpha) {
          if (score < beta) { alpha = score; }
          if constexpr (is_pv) { ss.prepend_to_pv(mv); }
        }
      }

      if (best_score >= beta || !internal.keep_going()) { break; }
    }
  }

  // Threat quiescence: explore a few quiets that create an immediate blast-mate threat
  // Only at the first qsearch level (elevation == 0) and if we are near the window
  if (!is_check_any && best_score < beta && elevation == 0 && best_score + 100 >= alpha && internal.keep_going()) {
    constexpr int threat_limit = 6;
    int explored_threats = 0;

    const chess::square_set enemy_king = bd.man_.them(bd.turn()).king();
    const chess::square_set king_zone = enemy_king.any() ? chess::board::capture_blast(enemy_king.item()) : chess::square_set{};

    const chess::move_list quiets = bd.generate_moves<chess::generation_mode::quiet_and_check>();
    for (const auto& mv : quiets) {
      if (mv.is_noisy()) { continue; }
      if (explored_threats >= threat_limit) { break; }
      // Cheap pre-filter: only consider moves that give check or step into the king zone
      const bool to_in_zone = king_zone.any() && king_zone.is_member(mv.to());
      if (!to_in_zone && !mv.is_castle_oo<chess::color::white>() && !mv.is_castle_ooo<chess::color::white>() &&
          !mv.is_castle_oo<chess::color::black>() && !mv.is_castle_ooo<chess::color::black>()) {
        // generate_moves already ensures legality; we just skip far quiets that don't touch the king zone unless they give check
        // we have no direct "gives check" flag here, so rely on zone filter
        if (!to_in_zone) { continue; }
      }

      const chess::board bd_threat = bd.forward(mv);
      if (!bd_threat.has_atomic_blast_capture()) { continue; }

      ++explored_threats;

      ss.set_played(mv);
      const auto updates = eval::buildUpdates(bd, bd_threat);
      nnue_state.update<true>(updates, bd_threat);
      const score_type score = -q_search<is_pv, use_tt>(ss.next(), nnue_state, bd_threat, -beta, -alpha, elevation + 1);
      nnue_state.pop();

      if (score > best_score) {
        best_score = score;
        best_move = mv;
        if (score > alpha) {
          if (score < beta) { alpha = score; }
          if constexpr (is_pv) { ss.prepend_to_pv(mv); }
        }
      }

      if (best_score >= beta || !internal.keep_going()) { break; }
    }
  }

  if (legal_count == 0 && is_check_any) { return ss.loss_score(); }
  if (legal_count == 0) { return value; }

  if (use_tt && internal.keep_going()) {
    const bound_type bound = best_score >= beta ? bound_type::lower : bound_type::upper;
    const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, 0);
    external.tt->insert(entry);
  }

  return best_score;
}

template <bool is_pv, bool is_root>
pv_search_result_t<is_root> search_worker::pv_search(
    const stack_view& ss,
    eval::NnueState& nnue_state,
    const chess::board& bd,
    score_type alpha,
    const score_type& beta,
    depth_type depth,
    const chess::player_type& reducer) noexcept {
  static_assert(!is_root || is_pv);

  auto make_result = [](const score_type& score, const chess::move& mv) {
    if constexpr (is_root) { return pv_search_result_t<is_root>{score, mv}; }
    if constexpr (!is_root) { return score; }
  };

  // callback on entering search function
  const bool should_update = internal.keep_going() && (is_root || internal.one_of<nodes_per_update>());
  if (should_update) { external.on_update(*this); }

  // step 1. drop into qsearch if depth reaches zero
  if (depth <= 0) { return make_result(q_search<is_pv>(ss, nnue_state, bd, alpha, beta, 0), chess::move::null()); }
  ++internal.nodes;
  if (!bd.man_.us(bd.turn()).king().any()) { return make_result(ss.loss_score(), chess::move::null()); }
  if (!bd.man_.them(bd.turn()).king().any()) { return make_result(ss.win_score(), chess::move::null()); }

  // step 2. check if node is terminal
  const bool is_check = bd.is_check();
  const bool atomic_check = bd.in_atomic_blast_check();
  const bool is_check_any = is_check || atomic_check;

  if (!is_root && bd.is_trivially_drawn()) { return make_result(draw_score, chess::move::null()); }
  if (!is_root && bd.is_rule50_draw() && (!is_check_any || bd.generate_moves<chess::generation_mode::all>().size() != 0)) {
    return make_result(draw_score, chess::move::null());
  }

  if (!is_root && ss.upcoming_cycle_exists(bd)) {
    if (draw_score >= beta) { return make_result(draw_score, chess::move::null()); }
    alpha = std::max(draw_score, alpha);
  }

  if constexpr (is_root) {
    if (const syzygy::tb_dtz_result result = syzygy::probe_dtz(bd); result.success) { return make_result(result.score, result.move); }
  }

  const std::optional<transposition_table_entry> maybe = !ss.has_excluded() ? external.tt->find(bd.hash()) : std::nullopt;
  if (maybe.has_value()) {
    const transposition_table_entry entry = maybe.value();
    const bool is_cutoff = !is_pv && entry.depth() >= depth &&
                           ((entry.bound() == bound_type::lower && entry.score() >= beta) || entry.bound() == bound_type::exact ||
                            (entry.bound() == bound_type::upper && entry.score() <= alpha));
    if (is_cutoff) { return make_result(entry.score(), entry.best_move()); }
  }

  const score_type original_alpha = alpha;
  const bool tt_pv = is_pv || (maybe.has_value() && maybe->tt_pv());

  if (const syzygy::tb_wdl_result result = syzygy::probe_wdl(bd); !is_root && result.success) {
    ++internal.tb_hits;

    switch (result.wdl) {
      case syzygy::wdl_type::loss: return make_result(ss.loss_score(), chess::move::null());
      case syzygy::wdl_type::draw: return make_result(draw_score, chess::move::null());
      case syzygy::wdl_type::win: return make_result(ss.win_score(), chess::move::null());
    }
  }

  // step 3. internal iterative reductions
  const bool should_iir = !maybe.has_value() && !ss.has_excluded() && depth >= external.constants->iir_depth();
  if (should_iir) { --depth; }

  // step 4. compute static eval and adjust appropriately if there's a tt hit
  const auto [feature_hash, static_value, value] = evaluate<is_pv>(ss, nnue_state, bd, maybe);

  // step 5. return static eval if max depth was reached
  if (ss.reached_max_height()) { return make_result(value, chess::move::null()); }

  // step 6. add position and static eval to stack
  ss.set_hash(bd.sided_hash()).set_eval(static_value);
  const bool improving = !is_check_any && ss.improving();
  const chess::square_set threatened = bd.them_threat_mask();

  const bool try_razor = !is_pv && !is_check_any && !ss.has_excluded() && depth <= external.constants->razor_depth() &&
                         value + external.constants->razor_margin(depth) <= alpha;

  if (try_razor) {
    const score_type razor_score = q_search<false>(ss, nnue_state, bd, alpha, alpha + 1, 0);
    if (razor_score <= alpha) { return make_result(razor_score, chess::move::null()); }
  }

  // step 7. static null move pruning
  const bool snm_prune = !is_pv && !ss.has_excluded() && !is_check_any && depth <= external.constants->snmp_depth() &&
                         value > beta + external.constants->snmp_margin(improving, threatened.any(), depth) && value > ss.loss_score();

  if (snm_prune) {
    const score_type adjusted_value = (beta + value) / 2;
    return make_result(adjusted_value, chess::move::null());
  }

  // step 8. null move pruning
  const bool try_nmp =
      !is_pv && !ss.has_excluded() && !is_check_any && depth >= external.constants->nmp_depth() && value > beta && ss.nmp_valid() &&
      bd.has_non_pawn_material() && (!threatened.any() || depth >= 4) &&
      (!maybe.has_value() || (maybe->bound() == bound_type::lower && bd.is_legal<chess::generation_mode::all>(maybe->best_move()) &&
                              !bd.see_gt(maybe->best_move(), external.constants->nmp_see_threshold())));

  if (try_nmp) {
    ss.set_played(chess::move::null());
    const depth_type adjusted_depth = std::max(0, depth - external.constants->nmp_reduction(depth, beta, value));
    const chess::board bd_null = bd.forward(chess::move::null());
    const auto updates = eval::buildUpdates(bd, bd_null);
    nnue_state.update<true>(updates, bd_null);
    const score_type nmp_score =
        -pv_search<false>(ss.next(), nnue_state, bd_null, -beta, -beta + 1, adjusted_depth, chess::player_from(!bd.turn()));
    nnue_state.pop();
    if (nmp_score >= beta) { return make_result(nmp_score, chess::move::null()); }
  }

  // step 9. probcut pruning
  const depth_type probcut_depth = external.constants->probcut_search_depth(depth);
  const score_type probcut_beta = external.constants->probcut_beta(beta);
  const bool try_probcut = !is_pv && !ss.has_excluded() && depth >= external.constants->probcut_depth() &&
                           !(maybe.has_value() && maybe->best_move().is_quiet()) &&
                           !(maybe.has_value() && maybe->depth() >= probcut_depth && maybe->score() < probcut_beta);

  if (try_probcut) {
    move_orderer<chess::generation_mode::noisy_and_check> probcut_orderer(move_orderer_data(&bd, &internal.hh.us(bd.turn())));
    if (maybe.has_value()) { probcut_orderer.set_first(maybe->best_move()); }

    for (const auto& [idx, mv] : probcut_orderer) {
      if (!internal.keep_going()) { break; }
      if (mv == ss.excluded()) { continue; }
      if (!bd.see_ge(mv, 0)) { continue; }

      ss.set_played(mv);

      const chess::board bd_ = bd.forward(mv);
      external.tt->prefetch(bd_.hash());
      internal.cache.prefetch(bd_.hash());
      const auto updates = eval::buildUpdates(bd, bd_);
      nnue_state.update<true>(updates, bd_);
      auto pv_score = [&] { return -pv_search<false>(ss.next(), nnue_state, bd_, -probcut_beta, -probcut_beta + 1, probcut_depth, reducer); };
      const score_type q_score = -q_search<false>(ss.next(), nnue_state, bd_, -probcut_beta, -probcut_beta + 1, 0);
      nnue_state.pop();
      const score_type probcut_score = (q_score >= probcut_beta) ? pv_score() : q_score;

      if (probcut_score >= probcut_beta) { return make_result(probcut_score, mv); }
    }
  }

  // step 10. initialize move orderer (setting tt move first if applicable)
  const chess::move killer = ss.killer();
  const chess::move follow = ss.follow();
  const chess::move counter = ss.counter();
  const zobrist::hash_type pawn_hash = bd.pawn_hash();

  move_orderer<chess::generation_mode::all> orderer(move_orderer_data(&bd, &internal.hh.us(bd.turn()))
                                                        .set_killer(killer)
                                                        .set_follow(follow)
                                                        .set_counter(counter)
                                                        .set_threatened(threatened)
                                                        .set_pawn_hash(pawn_hash));

  if (maybe.has_value()) { orderer.set_first(maybe->best_move()); }

  // list of attempted moves for updating histories
  chess::move_list moves_tried{};

  // move loop
  score_type best_score = ss.loss_score();
  chess::move best_move = chess::move::null();

  bool did_double_extend{false};
  int legal_count{0};

  for (const auto& [idx, mv] : orderer) {
    ++legal_count;
    if (!internal.keep_going()) { break; }
    if (mv == ss.excluded()) { continue; }

    const std::size_t nodes_before = internal.nodes.load(std::memory_order_relaxed);
    const counter_type history_value = internal.hh.us(bd.turn()).compute_value(history::context{follow, counter, threatened, pawn_hash}, mv);

    if (bd.is_atomic_king_blast_capture(mv)) {
      ss.set_played(mv);
      const score_type score = ss.win_score();

      if (score > best_score) {
        best_score = score;
        best_move = mv;
        if (score > alpha) {
          alpha = score;
          if constexpr (is_pv) { ss.prepend_to_pv(mv); }
        }
      }

      if constexpr (is_root) { internal.node_distribution[mv] += (internal.nodes.load(std::memory_order_relaxed) - nodes_before); }

      if (best_score >= beta) { break; }
      continue;
    }

    const chess::board bd_ = bd.forward(mv);

    const bool try_pruning = !is_root && idx >= 2 && best_score > max_mate_score;

    // step 11. pruning
    if (try_pruning) {
      const bool child_check_any = bd_.is_check() || bd_.in_atomic_blast_check();
      const bool lm_prune =
          !child_check_any && depth <= external.constants->lmp_depth() && idx > external.constants->lmp_count(improving, depth);

      if (lm_prune) { break; }

      const bool futility_prune =
          mv.is_quiet() && depth <= external.constants->futility_prune_depth() && value + external.constants->futility_margin(depth) < alpha;

      if (futility_prune) { continue; }

      const bool quiet_see_prune = mv.is_quiet() && depth <= external.constants->quiet_see_prune_depth() &&
                                   !bd.see_ge(mv, external.constants->quiet_see_prune_threshold(depth));

      if (quiet_see_prune) { continue; }

      const bool noisy_see_prune = mv.is_noisy() && depth <= external.constants->noisy_see_prune_depth() &&
                                   !bd.see_ge(mv, external.constants->noisy_see_prune_threshold(depth));

      if (noisy_see_prune) { continue; }

      const bool history_prune = mv.is_quiet() && history_value <= external.constants->history_prune_threshold(depth);

      if (history_prune) { continue; }
    }

    external.tt->prefetch(bd_.hash());
    internal.cache.prefetch(bd_.hash());

    // step 12. extensions
    bool multicut = false;
    const depth_type extension = [&, mv = mv] {
      const bool try_singular = !is_root && !ss.has_excluded() && depth >= external.constants->singular_extension_depth() && maybe.has_value() &&
                                mv == maybe->best_move() && maybe->bound() != bound_type::upper &&
                                maybe->depth() + external.constants->singular_extension_depth_margin() >= depth;

      if (try_singular) {
        const depth_type singular_depth = external.constants->singular_search_depth(depth);
        const score_type singular_beta = external.constants->singular_beta(maybe->score(), depth);
        ss.set_excluded(mv);
        const score_type excluded_score = pv_search<false>(ss, nnue_state, bd, singular_beta - 1, singular_beta, singular_depth, reducer);
        ss.set_excluded(chess::move::null());

        if (!is_pv && excluded_score + external.constants->singular_double_extension_margin() < singular_beta) {
          did_double_extend = true;
          return 2;
        }

        if (excluded_score < singular_beta) { return 1; }
        if (excluded_score >= beta) { multicut = true; }
        if constexpr (!is_pv) { return -1; }
      }

      return 0;
    }();

    if (!is_root && multicut) { return make_result(beta, chess::move::null()); }

    ss.set_played(mv);
    const auto updates = eval::buildUpdates(bd, bd_);
    nnue_state.update<true>(updates, bd_);

    const score_type score = [&, this, idx = idx, mv = mv] {
      const depth_type next_depth = depth + extension - 1;

      auto full_width = [&] { return -pv_search<is_pv>(ss.next(), nnue_state, bd_, -beta, -alpha, next_depth, reducer); };

      auto zero_width = [&](const depth_type& zw_depth) {
        const chess::player_type next_reducer = (is_pv || zw_depth < next_depth) ? chess::player_from(bd.turn()) : reducer;
        return -pv_search<false>(ss.next(), nnue_state, bd_, -alpha - 1, -alpha, zw_depth, next_reducer);
      };

      if (is_pv && idx == 0) { return full_width(); }

      depth_type lmr_depth;
      score_type zw_score;

      // step 13. late move reductions
      const bool try_lmr = !is_check_any && (mv.is_quiet() || !bd.see_ge(mv, 0)) && idx >= 2 && (depth >= external.constants->reduce_depth());
      if (try_lmr) {
        depth_type reduction = external.constants->reduction(depth, idx);

        // adjust reduction
        if (mv.piece() == chess::piece_type::pawn) {
          const int to_rank = mv.to().index() / 8;
          const bool near_promo = bd.turn() ? (to_rank == 6) : (to_rank == 1);
          if (mv.is_promotion() || near_promo) { reduction = 0; }
        }
        if (improving) { --reduction; }
        if (bd_.is_check() || bd_.in_atomic_blast_check()) { --reduction; }
        if (bd.creates_threat(mv)) { --reduction; }
        if (mv == killer) { --reduction; }

        if (!tt_pv) { ++reduction; }
        if (did_double_extend) { ++reduction; }

        // if our opponent is the reducing player, an errant fail low will, at worst, induce a re-search
        // this idea is at least similar (maybe equivalent) to the "cutnode idea" found in Stockfish.
        if (is_player(reducer, !bd.turn())) { ++reduction; }

        if (mv.is_quiet()) { reduction += external.constants->history_reduction(history_value); }

        reduction = std::max(0, reduction);

        lmr_depth = std::max(1, next_depth - reduction);
        zw_score = zero_width(lmr_depth);
      }

      // search again at full depth if necessary
      if (!try_lmr || (zw_score > alpha && lmr_depth < next_depth)) { zw_score = zero_width(next_depth); }

      // search again with full window on pv nodes
      return (is_pv && (alpha < zw_score && zw_score < beta)) ? full_width() : zw_score;
    }();
    nnue_state.pop();

    if (score < beta && (mv.is_quiet() || !bd.see_gt(mv, 0))) { moves_tried.push(mv); }

    if (score > best_score) {
      best_score = score;
      best_move = mv;
      if (score > alpha) {
        if (score < beta) { alpha = score; }
        if constexpr (is_pv) { ss.prepend_to_pv(mv); }
      }
    }

    if constexpr (is_root) { internal.node_distribution[mv] += (internal.nodes.load(std::memory_order_relaxed) - nodes_before); }

    if (best_score >= beta) { break; }
  }

  if (legal_count == 0 && is_check_any) { return make_result(ss.loss_score(), chess::move::null()); }
  if (legal_count == 0) { return make_result(draw_score, chess::move::null()); }

  // step 14. update histories if appropriate and maybe insert a new transposition_table_entry
  if (internal.keep_going() && !ss.has_excluded()) {
    const bound_type bound = [&] {
      if (best_score >= beta) { return bound_type::lower; }
      if (is_pv && best_score > original_alpha) { return bound_type::exact; }
      return bound_type::upper;
    }();

    if (bound == bound_type::lower && (best_move.is_quiet() || !bd.see_gt(best_move, 0))) {
      internal.hh.us(bd.turn()).update(history::context{follow, counter, threatened, pawn_hash}, best_move, moves_tried, depth);
      ss.set_killer(best_move);
    }

    if (!is_check_any && best_move.is_quiet()) {
      const score_type error = best_score - static_value;
      internal.correction.us(bd.turn()).update(feature_hash, bound, error, depth);
    }

    const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, depth, tt_pv);
    external.tt->insert(entry);
  }

  return make_result(best_score, best_move);
}

void search_worker::iterative_deepening_loop() noexcept {
  internal.nnue_state.reset(internal.stack.root());

  score_type alpha = -big_number;
  score_type beta = big_number;
  for (; internal.keep_going(); ++internal.depth) {
    internal.depth = std::min(max_depth, internal.depth.load());
    // update aspiration window once reasonable evaluation is obtained
    if (internal.depth >= external.constants->aspiration_depth()) {
      const score_type previous_score = internal.score;
      alpha = previous_score - aspiration_delta;
      beta = previous_score + aspiration_delta;
    }

    score_type delta = aspiration_delta;
    depth_type consecutive_failed_high_count{0};

    for (;;) {
      internal.stack.clear_future();

      const depth_type adjusted_depth = std::max(1, internal.depth - consecutive_failed_high_count);
      const auto [search_score, search_move] = pv_search<true, true>(
          stack_view::root(internal.stack), internal.nnue_state, internal.stack.root(), alpha, beta, adjusted_depth, chess::player_type::none);

      if (!internal.keep_going()) { break; }

      // update aspiration window if failing low or high
      if (search_score <= alpha) {
        beta = (alpha + beta) / 2;
        alpha = search_score - delta;
        consecutive_failed_high_count = 0;
      } else if (search_score >= beta) {
        beta = search_score + delta;
        ++consecutive_failed_high_count;
      } else {
        // store updated information
        internal.score.store(search_score);
        if (!search_move.is_null()) {
          internal.best_move.store(search_move.data);
          internal.ponder_move.store(internal.stack.ponder_move().data);
        }
        break;
      }

      // exponentially grow window
      delta += delta / 3;
    }

    // callback on iteration completion
    if (internal.keep_going()) { external.on_iter(*this); }
  }
}

}  // namespace search

template search::score_type search::search_worker::q_search<false, false>(
    const stack_view& ss,
    eval::NnueState& nnue_state,
    const chess::board& bd,
    score_type alpha,
    const score_type& beta,
    const depth_type& elevation);

template search::score_type search::search_worker::q_search<true, false>(
    const stack_view& ss,
    eval::NnueState& nnue_state,
    const chess::board& bd,
    score_type alpha,
    const score_type& beta,
    const depth_type& elevation);
