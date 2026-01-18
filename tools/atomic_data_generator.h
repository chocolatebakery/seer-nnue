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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <atomic_data_writer.h>
#include <chess/board_history.h>
#include <eval/nnue.h>
#include <search/search_constants.h>
#include <search/search_worker.h>

namespace train {

using state_type = chess::board;
using score_type = search::score_type;

enum class result_type { win, draw, loss };

enum class atomic_filter_preset { minimal, balanced, quiet };

struct seed_provider {
  virtual ~seed_provider() = default;
  virtual bool next(state_type& out, std::mt19937_64& gen) = 0;
};

inline bool kings_present(const state_type& state) {
  return state.man_.white.king().count() == 1 && state.man_.black.king().count() == 1;
}

inline result_type mirrored_result(const result_type& result) {
  switch (result) {
    case result_type::win: return result_type::loss;
    case result_type::loss: return result_type::win;
    case result_type::draw: return result_type::draw;
  }
  return result_type::draw;
}

inline bool has_repetition(const chess::board_history& hist, const state_type& state) {
  const auto key = state.sided_hash();
  const std::size_t sz = hist.size();
  for (std::size_t i = 0; i < sz; ++i) {
    if (hist.at(i).white == key.white && hist.at(i).black == key.black) { return true; }
  }
  return false;
}

inline bool is_terminal(const chess::board_history& hist, const state_type& state) {
  if (has_repetition(hist, state)) { return true; }
  if (!state.man_.us(state.turn()).king().any()) { return true; }
  if (!state.man_.them(state.turn()).king().any()) { return true; }
  if (state.generate_moves().size() == 0) { return true; }
  return false;
}

inline result_type get_result(const chess::board_history& hist, const state_type& state) {
  if (has_repetition(hist, state)) { return result_type::draw; }
  if (!state.man_.us(state.turn()).king().any()) { return result_type::loss; }
  if (!state.man_.them(state.turn()).king().any()) { return result_type::win; }

  if (state.generate_moves().size() == 0) {
    return state.is_check() ? result_type::loss : result_type::draw;
  }

  return result_type::draw;
}

inline result_type get_result_with_adjudication(
    const chess::board_history& hist, const state_type& state, const score_type& final_score) {
  const result_type terminal_result = get_result(hist, state);
  if (terminal_result != result_type::draw) { return terminal_result; }

  constexpr score_type adjudication_threshold = 1000;
  if (final_score >= adjudication_threshold) { return result_type::win; }
  if (final_score <= -adjudication_threshold) { return result_type::loss; }
  return result_type::draw;
}

inline Outcome outcome_from_white_result(const result_type& white_result) {
  switch (white_result) {
    case result_type::win: return Outcome::WhiteWin;
    case result_type::loss: return Outcome::WhiteLoss;
    case result_type::draw: return Outcome::Draw;
  }
  return Outcome::Draw;
}

struct dedup_cache {
  std::mutex mutex_;
  std::unordered_set<std::uint64_t> seen_;
  std::deque<std::uint64_t> order_;
  std::size_t capacity_{0};

  bool accept(const std::uint64_t key) {
    if (capacity_ == 0) { return true; }
    std::lock_guard<std::mutex> lck(mutex_);
    if (seen_.find(key) != seen_.end()) { return false; }
    seen_.insert(key);
    order_.push_back(key);
    if (order_.size() > capacity_) {
      const auto old = order_.front();
      order_.pop_front();
      seen_.erase(old);
    }
    return true;
  }

  explicit dedup_cache(const std::size_t& capacity) : capacity_{capacity} { seen_.reserve(capacity); }
};

struct atomic_data_generator {
  std::size_t concurrency_{1};

  search::depth_type ply_limit_{256};
  search::depth_type random_ply_min_{10};
  search::depth_type random_ply_max_{10};
  search::depth_type fixed_depth_{6};
  std::size_t fixed_nodes_{5120};

  score_type eval_limit_{6144};
  std::size_t min_pieces_{0};
  double require_capture_prob_{0.0};
  atomic_filter_preset filter_{atomic_filter_preset::balanced};
  bool quiet_filter_enabled_{true};
  bool allow_mate_in_one_{false};
  std::uint64_t seed_{1};
  std::uint64_t progress_every_{0};

  std::shared_ptr<search::transposition_table> tt_{nullptr};
  std::shared_ptr<search::search_constants> constants_ = std::make_shared<search::search_constants>(1);
  std::shared_ptr<dedup_cache> dedup_{nullptr};

  std::shared_ptr<seed_provider> seed_provider_{nullptr};
  std::vector<state_type> seeds_{};
  atomic_data_writer writer_;

  atomic_data_generator& set_concurrency(const std::size_t& concurrency) {
    concurrency_ = std::max<std::size_t>(1, concurrency);
    constants_->update_(concurrency_);
    return *this;
  }

  atomic_data_generator& set_fixed_depth(const search::depth_type& fixed_depth) {
    fixed_depth_ = fixed_depth;
    return *this;
  }

  atomic_data_generator& set_fixed_nodes(const std::size_t& fixed_nodes) {
    fixed_nodes_ = fixed_nodes;
    return *this;
  }

  atomic_data_generator& set_ply_limit(const search::depth_type& ply_limit) {
    ply_limit_ = ply_limit;
    return *this;
  }

  atomic_data_generator& set_random_ply_range(const search::depth_type& min_ply, const search::depth_type& max_ply) {
    random_ply_min_ = std::max<search::depth_type>(0, min_ply);
    random_ply_max_ = std::max(random_ply_min_, max_ply);
    return *this;
  }

  atomic_data_generator& set_eval_limit(const score_type& eval_limit) {
    eval_limit_ = std::abs(eval_limit);
    return *this;
  }

  atomic_data_generator& set_min_pieces(const std::size_t& min_pieces) {
    min_pieces_ = min_pieces;
    return *this;
  }

  atomic_data_generator& set_require_capture_prob(const double& prob) {
    require_capture_prob_ = std::clamp(prob, 0.0, 1.0);
    return *this;
  }

  atomic_data_generator& set_filter(const atomic_filter_preset& filter) {
    filter_ = filter;
    return *this;
  }

  atomic_data_generator& set_quiet_filter_enabled(const bool& enabled) {
    quiet_filter_enabled_ = enabled;
    return *this;
  }

  atomic_data_generator& set_allow_mate_in_one(const bool& allow) {
    allow_mate_in_one_ = allow;
    return *this;
  }

  atomic_data_generator& set_seed(const std::uint64_t& seed) {
    seed_ = seed;
    return *this;
  }

  atomic_data_generator& set_seed_provider(const std::shared_ptr<seed_provider>& provider) {
    seed_provider_ = provider;
    return *this;
  }

  atomic_data_generator& set_progress_every(const std::uint64_t& progress_every) {
    progress_every_ = progress_every;
    return *this;
  }

  atomic_data_generator& set_dedup_capacity(const std::size_t& capacity) {
    if (capacity == 0) {
      dedup_.reset();
    } else {
      dedup_ = std::make_shared<dedup_cache>(capacity);
    }
    return *this;
  }

  atomic_data_generator& set_seeds(const std::vector<state_type>& seeds) {
    seeds_ = seeds;
    return *this;
  }

  atomic_data_generator& add_seed(const state_type& seed) {
    seeds_.push_back(seed);
    return *this;
  }

  atomic_data_generator& generate_data() {
    constexpr score_type nnue_scale_numerator = 1024;
    constexpr score_type nnue_scale_denominator = 288;

    const std::vector<state_type> seed_pool = seeds_;

    writer_.set_progress_every(progress_every_);

    auto generate = [&, this](const std::size_t thread_idx) {
      using worker_type = search::search_worker;
      const std::uint64_t mix = 0x9e3779b97f4a7c15ULL * (thread_idx + 1);
      std::mt19937_64 gen(seed_ ^ mix);
      std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
      std::optional<std::uniform_int_distribution<std::size_t>> seed_dist{};
      if (!seed_pool.empty()) { seed_dist = std::uniform_int_distribution<std::size_t>(0, seed_pool.size() - 1); }

      auto next_seed = [&]() -> state_type {
        if (seed_provider_) {
          state_type seed{};
          if (seed_provider_->next(seed, gen)) { return seed; }
        }
        if (seed_dist.has_value()) { return seed_pool[(*seed_dist)(gen)]; }
        return state_type::start_pos();
      };

      std::unique_ptr<worker_type> worker = std::make_unique<worker_type>(
          tt_,
          constants_,
          [&, this](const auto& w) {
            if (w.depth() >= fixed_depth_) { worker->stop(); }
          },
          [&, this](const auto& w) {
            if (w.nodes() >= fixed_nodes_) { worker->stop(); }
          });

      while (!writer_.is_complete()) {
        (worker->internal).reset();

        std::vector<datagen_entry> block{};

        chess::board_history hist{};
        state_type state = next_seed();

        const search::depth_type min_random = std::min(random_ply_min_, ply_limit_);
        const search::depth_type max_random = std::min(random_ply_max_, ply_limit_);
        const search::depth_type lo = std::min(min_random, max_random);
        const search::depth_type hi = std::max(min_random, max_random);
        const search::depth_type random_ply =
            (lo == hi) ? lo : std::uniform_int_distribution<search::depth_type>(lo, hi)(gen);

        score_type final_score = 0;
        const result_type game_result = [&] {
          for (search::depth_type ply = 0; ply <= ply_limit_ && !is_terminal(hist, state); ++ply) {
            if (ply < random_ply) {
              const auto mv_list = state.generate_moves();
              if (mv_list.empty()) { break; }
              const std::size_t idx = std::uniform_int_distribution<std::size_t>(0, mv_list.size() - 1)(gen);

              hist.push(state.sided_hash());
              state = state.forward(mv_list[idx]);
              continue;
            }

            worker->go(hist, state, 1);
            worker->iterative_deepening_loop();
            worker->stop();

            const auto best_move = worker->best_move();
            const auto pov_score = worker->score();
            final_score = pov_score;

            const bool has_kings = kings_present(state);
            const bool enough_pieces = min_pieces_ == 0 || state.num_pieces() >= min_pieces_;
            if (has_kings && enough_pieces) {
              const auto captures = state.generate_moves<chess::generation_mode::noisy>();
              bool mate_in_one = false;
              for (const auto& mv : captures) {
                if (state.is_atomic_king_blast_capture(mv)) {
                  mate_in_one = true;
                  break;
                }
              }

              if (!mate_in_one || allow_mate_in_one_) {
                const bool direct_check = state.is_check();
                const bool atomic_check = state.in_atomic_blast_check();
                const bool has_capture = !captures.empty();
                const bool contact = direct_check || atomic_check || has_capture;

                const bool require_contact = (require_capture_prob_ > 0.0) && (prob_dist(gen) < require_capture_prob_);

                bool accept = !require_contact || contact;
                if (accept && filter_ == atomic_filter_preset::quiet && quiet_filter_enabled_) {
                  accept = false;
                  if (!direct_check && !atomic_check) {
                    const chess::color stm = state.turn() ? chess::color::white : chess::color::black;
                    const auto static_cp = eval::NnueState::evaluateOnce(state, stm);
                    const score_type static_eval = static_cast<score_type>(static_cp) * nnue_scale_numerator / nnue_scale_denominator;

                    worker->go(hist, state, 1);
                    eval::NnueState q_state{};
                    q_state.reset(state);
                    const auto view = search::stack_view::root((worker->internal).stack);
                    const score_type q_eval =
                        worker->q_search<true, false>(view, q_state, state, search::mate_score, -search::mate_score, 0);
                    worker->stop();

                    if (static_eval == q_eval) { accept = true; }
                  }
                }

                if (accept && dedup_ && !dedup_->accept(state.hash())) { accept = false; }

                if (accept) {
                  const score_type white_score = state.turn() ? pov_score : -pov_score;
                  block.push_back(datagen_entry{state, white_score});
                }
              }
            }

            hist.push(state.sided_hash());
            state = state.forward(best_move);
          }

          return get_result_with_adjudication(hist, state, final_score);
        }();

        const result_type white_result = state.turn() ? game_result : mirrored_result(game_result);
        writer_.write_block(block, outcome_from_white_result(white_result));
      }
    };

    std::vector<std::thread> threads{};
    for (std::size_t i = 0; i < concurrency_; ++i) { threads.emplace_back(generate, i); }
    for (auto& thread : threads) { thread.join(); }
    return *this;
  }

  atomic_data_generator(const std::string& path, const std::size_t& total, const std::size_t& tt_mb_size)
      : writer_{path, total} {
    tt_ = std::make_shared<search::transposition_table>(tt_mb_size);
  }
};

}  // namespace train
