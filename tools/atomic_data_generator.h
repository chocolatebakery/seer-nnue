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

#include <chess/board_history.h>
#include <nnue/eval.h>
#include <nnue/weights_streamer.h>
#include <search/search_constants.h>
#include <search/search_worker.h>

#include <atomic_data_writer.h>
#define NNUE_EMBEDDED_WEIGHTS_EXTERN
#include <sample.h>
#undef NNUE_EMBEDDED_WEIGHTS_EXTERN

namespace train {

enum class atomic_filter_preset { minimal, balanced, quiet };

struct seed_provider {
  virtual ~seed_provider() = default;
  virtual bool next(state_type& out, std::mt19937_64& gen) = 0;
};

inline bool kings_present(const state_type& state) {
  return state.man_.white.king().count() == 1 && state.man_.black.king().count() == 1;
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
    const nnue::quantized_weights q_weights = [&, this] {
      nnue::weights tmp{};
      nnue::embedded_weight_streamer embedded(nnue::embed::weights_file_data);
      tmp.load(embedded);
      return tmp.to<nnue::quantized_weights>();
    }();

    const std::vector<state_type> seed_pool = seeds_;

    writer_.set_progress_every(progress_every_);

    auto generate = [&, this](const std::size_t thread_idx) {
      using worker_type = search::search_worker;
      const std::uint64_t mix = 0x9e3779b97f4a7c15ULL * (thread_idx + 1);
      std::mt19937_64 gen(seed_ ^ mix);
      std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
      std::optional<std::uniform_int_distribution<std::size_t>> seed_dist{};
      if (!seed_pool.empty()) {
        seed_dist = std::uniform_int_distribution<std::size_t>(0, seed_pool.size() - 1);
      }

      auto next_seed = [&]() -> state_type {
        if (seed_provider_) {
          state_type seed{};
          if (seed_provider_->next(seed, gen)) { return seed; }
        }
        if (seed_dist.has_value()) { return seed_pool[(*seed_dist)(gen)]; }
        return state_type::start_pos();
      };

      std::unique_ptr<worker_type> worker = std::make_unique<worker_type>(
          &q_weights, tt_, constants_,
          [&, this](const auto& w) {
            if (w.depth() >= fixed_depth_) { worker->stop(); }
          },
          [&, this](const auto& w) {
            if (w.nodes() >= fixed_nodes_) { worker->stop(); }
          });

      while (!writer_.is_complete()) {
        (worker->internal).reset();

        std::vector<sample> block{};

        chess::board_history hist{};
        state_type state = next_seed();

        const search::depth_type min_random = std::min(random_ply_min_, ply_limit_);
        const search::depth_type max_random = std::min(random_ply_max_, ply_limit_);
        const search::depth_type lo = std::min(min_random, max_random);
        const search::depth_type hi = std::max(min_random, max_random);
        const search::depth_type random_ply = (lo == hi)
                                                  ? lo
                                                  : std::uniform_int_distribution<search::depth_type>(lo, hi)(gen);

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
            const auto best_score = worker->score();

            if (best_score >= eval_limit_) { return result_type::win; }
            if (best_score <= -eval_limit_) { return result_type::loss; }

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
                    const auto view = search::stack_view::root((worker->internal).stack);
                    const auto evaluator = [&] {
                      nnue::eval result(&q_weights, &worker->internal.scratchpad, 0, 0);
                      state.feature_full_reset(result);
                      return result;
                    }();

                    const auto static_eval_data = evaluator.evaluate(state.turn(), state.phase<real_type>());
                    const search::score_type static_eval = static_eval_data.result;

                    worker->go(hist, state, 1);
                    nnue::eval_node node = nnue::eval_node::clean_node(evaluator);
                    const search::score_type q_eval =
                        worker->q_search<true, false>(view, node, state, search::mate_score, -search::mate_score, 0);
                    worker->stop();

                    if (static_eval == q_eval) { accept = true; }
                  }
                }

                if (accept && dedup_ && !dedup_->accept(state.hash())) { accept = false; }

                if (accept) { block.emplace_back(state, best_score); }
              }
            }

            hist.push(state.sided_hash());
            state = state.forward(best_move);
          }

          return get_result(hist, state);
        }();

        for (auto& elem : block) { elem.set_result(relative_result(state.turn(), elem.pov(), game_result)); }

        writer_.write_block(block);
      }
    };

    std::vector<std::thread> threads{};
    for (std::size_t i = 0; i < concurrency_; ++i) { threads.emplace_back(generate, i); }
    for (auto& thread : threads) { thread.join(); }
    return *this;
  }

  atomic_data_generator(const std::string& path, const std::size_t& total, const std::size_t& tt_mb_size) : writer_{path, total} {
    tt_ = std::make_shared<search::transposition_table>(tt_mb_size);
  }
};

}  // namespace train
