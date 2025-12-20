#pragma once

#include <chess/types.h>
#include <feature/util.h>
#include <chess/move.h>
#include <chess/board_history.h>
#include <nnue/eval.h>
#include <search/search_constants.h>
#include <search/search_worker.h>
#include <nnue/embedded_weights.h>
#include <nnue/weights_streamer.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>

namespace train {

using real_type = float;
using state_type = chess::board;
using score_type = search::score_type;

enum class result_type {
  win,
  draw,
  loss,
  undefined
};

constexpr char result_to_char(const result_type& result){
  switch(result){
    case result_type::win: return 'w';
    case result_type::draw: return 'd';
    case result_type::loss: return 'l';
    default: return 'u';
  }
}

constexpr result_type result_from_char(const char& result){
  switch(result){
    case 'w': return result_type::win;
    case 'd': return result_type::draw;
    case 'l': return result_type::loss;
    default: return result_type::undefined;
  }
}

constexpr result_type mirrored_result(const result_type& result){
  switch(result){
    case result_type::win: return result_type::loss;
    case result_type::draw: return result_type::draw;
    case result_type::loss: return result_type::win;
    default: return result_type::undefined;
  }
}

constexpr size_t half_feature_numel(){ return feature::half_ka::numel; }
constexpr size_t max_active_half_features(){ return feature::half_ka::max_active_half_features; }

real_type sigmoid(const real_type& x) {
  constexpr real_type one = static_cast<real_type>(1);
  return one / (std::exp(-x) + one);
}

struct feature_set : chess::sided<feature_set, std::set<size_t>> {
  std::set<size_t> white;
  std::set<size_t> black;

  feature_set() : white{}, black{} {}
};

inline bool has_repetition(const chess::board_history& hist, const state_type& state) {
  const auto key = state.sided_hash();
  const std::size_t sz = hist.size();
  for (std::size_t i = 0; i < sz; ++i) {
    if (hist.at(i).white == key.white && hist.at(i).black == key.black) { return true; }
  }
  return false;
}

bool is_terminal(const chess::board_history& hist, const state_type& state) {
  if (has_repetition(hist, state)) { return true; }
  if (!state.man_.us(state.turn()).king().any()) { return true; }
  if (!state.man_.them(state.turn()).king().any()) { return true; }
  if (state.generate_moves().size() == 0) { return true; }
  return false;
}

result_type get_result(const chess::board_history& hist, const state_type& state) {
  if (has_repetition(hist, state)) { return result_type::draw; }
  if (!state.man_.us(state.turn()).king().any()) { return result_type::loss; }
  if (!state.man_.them(state.turn()).king().any()) { return result_type::win; }
  if (state.generate_moves().size() == 0) { return result_type::draw; }
  return result_type::draw;
}

result_type relative_result(const bool& pov_a, const bool& pov_b, const result_type& result){
  return pov_a == pov_b ? result : mirrored_result(result);
}

feature_set get_features(const state_type& state) {
  feature_set features{};
  state.feature_full_reset(features);
  return features;
}

std::tuple<std::vector<nnue::weights::parameter_type>, std::vector<nnue::weights::parameter_type>> feature_transformer_parameters() {
  // NOTE: In Seer, the embedded eval file is typically `quantized_weights` (compact, fast for search).
  // Python training wants float32 weights for the (frozen) feature transformer, so when the embedded
  // file is quantized we dequantize the shared layer back to float.
  const std::size_t embedded_bytes = static_cast<std::size_t>(nnue::embed::weights_file_size);

  // Expected file sizes for Seer NNUE formats (based on compile-time dims).
  constexpr std::size_t numel = feature::half_ka::numel;
  constexpr std::size_t base = nnue::weights::base_dim;

  // Float `nnue::weights` format: shared (float) + all dense layers (float). No signature prefix.
  constexpr std::size_t expected_float_bytes =
      (numel * base + base) * sizeof(nnue::weights::parameter_type) +
      ((2 * base) * 8 + 8) * sizeof(nnue::weights::parameter_type) +
      (8 * 8 + 8) * sizeof(nnue::weights::parameter_type) +
      (16 * 8 + 8) * sizeof(nnue::weights::parameter_type) +
      (24 * 1 + 1) * sizeof(nnue::weights::parameter_type);

  // Quantized `nnue::quantized_weights` format:
  // [u32 signature] + shared (i16) + fc0 (i8 weights, i32 bias) + fc1/2/3 (float).
  constexpr std::size_t expected_quantized_bytes =
      sizeof(nnue::weights_streamer::signature_type) +
      (numel * base + base) * sizeof(nnue::weights::quantized_parameter_type) +
      ((2 * base) * 8) * sizeof(nnue::weights::half_quantized_parameter_type) +
      (8) * sizeof(nnue::dot_type<nnue::weights::quantized_parameter_type>) +
      (8 * 8 + 8) * sizeof(nnue::weights::parameter_type) +
      (16 * 8 + 8) * sizeof(nnue::weights::parameter_type) +
      (24 * 1 + 1) * sizeof(nnue::weights::parameter_type);

  if (embedded_bytes == expected_quantized_bytes) {
    nnue::embedded_weight_streamer streamer(nnue::embed::weights_file_data);

    nnue::weights_streamer::signature_type signature{};
    streamer.stream(&signature);

    using q_feature_transformer_type =
        nnue::sparse_affine_layer<nnue::weights::quantized_parameter_type, feature::half_ka::numel, nnue::weights::base_dim>;
    q_feature_transformer_type ft{};
    ft.load_(streamer);

    std::vector<nnue::weights::parameter_type> weights{};
    weights.reserve(q_feature_transformer_type::W_numel);
    for (std::size_t i = 0; i < q_feature_transformer_type::W_numel; ++i) {
      weights.push_back(static_cast<nnue::weights::parameter_type>(ft.W[i]) / nnue::weights::shared_quantization_scale);
    }

    std::vector<nnue::weights::parameter_type> bias{};
    bias.reserve(q_feature_transformer_type::b_numel);
    for (std::size_t i = 0; i < q_feature_transformer_type::b_numel; ++i) {
      bias.push_back(static_cast<nnue::weights::parameter_type>(ft.b[i]) / nnue::weights::shared_quantization_scale);
    }

    return std::tuple(weights, bias);
  }

  if (embedded_bytes == expected_float_bytes) {
    nnue::embedded_weight_streamer streamer(nnue::embed::weights_file_data);
    using feature_transformer_type =
        nnue::sparse_affine_layer<nnue::weights::parameter_type, feature::half_ka::numel, nnue::weights::base_dim>;
    feature_transformer_type ft{};
    ft.load_(streamer);
    std::vector<nnue::weights::parameter_type> weights(ft.W, ft.W + feature_transformer_type::W_numel);
    std::vector<nnue::weights::parameter_type> bias(ft.b, ft.b + feature_transformer_type::b_numel);
    return std::tuple(weights, bias);
  }

  throw std::runtime_error(
      "Embedded eval weights have unexpected size: " + std::to_string(embedded_bytes) +
      " bytes. Expected quantized=" + std::to_string(expected_quantized_bytes) +
      " or float=" + std::to_string(expected_float_bytes) +
      ". Rebuild with -DEVALFILE=<path-to-eval.bin> pointing to a Seer NNUE weights file.");
}

}  // namespace train
