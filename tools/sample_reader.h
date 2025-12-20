#pragma once

#include <optional>
#include <string>
#include <array>
#include <cstdint>

#include <line_count_size.h>
#include <file_reader_iterator.h>
#include <sample.h>
#include <ios>

namespace train{

struct sample_reader : line_count_size<sample_reader> {
  using iterator = file_reader_iterator<sample>;
  std::string path_;
  bool binary_{false};

  static std::optional<sample> read_binary(std::ifstream& in) {
    std::uint8_t n{0};
    std::uint8_t stm{0};
    if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) { return std::nullopt; }
    if (!in.read(reinterpret_cast<char*>(&stm), sizeof(stm))) { return std::nullopt; }

    if (n < 2 || n > 32) { return std::nullopt; }

    static constexpr std::array<chess::color, 12> code_color = {
      chess::color::white, chess::color::white, chess::color::white, chess::color::white, chess::color::white, chess::color::white,
      chess::color::black, chess::color::black, chess::color::black, chess::color::black, chess::color::black, chess::color::black
    };

    static constexpr std::array<chess::piece_type, 6> code_piece = {
      chess::piece_type::pawn, chess::piece_type::knight, chess::piece_type::bishop,
      chess::piece_type::rook, chess::piece_type::queen, chess::piece_type::king
    };

    train::state_type board{};
    // turn() is determined by ply_count parity; even = white
    board.lat_.ply_count = (stm == 1) ? 0 : 1;
    board.lat_.half_clock = 0;
    board.lat_.white.set_oo(false);
    board.lat_.white.set_ooo(false);
    board.lat_.black.set_oo(false);
    board.lat_.black.set_ooo(false);
    board.lat_.white.clear_ep_mask();
    board.lat_.black.clear_ep_mask();

    std::uint8_t white_kings{0};
    std::uint8_t black_kings{0};

    for (std::uint8_t i = 0; i < n; ++i) {
      std::uint8_t code{0};
      std::uint8_t sq_idx{0};
      if (!in.read(reinterpret_cast<char*>(&code), sizeof(code))) { return std::nullopt; }
      if (!in.read(reinterpret_cast<char*>(&sq_idx), sizeof(sq_idx))) { return std::nullopt; }
      if (code >= code_color.size() || sq_idx >= 64) { return std::nullopt; }

      const chess::color c = code_color[code];
      const chess::piece_type pt = code_piece[code % 6];

      const int file = static_cast<int>(sq_idx % 8);
      const int rank = static_cast<int>(sq_idx / 8);
      // Internal board uses flipped file compared to python-chess indexing.
      const chess::tbl_square sq{7 - file, rank};
      board.man_.us(c).add_piece(pt, sq);

      if (pt == chess::piece_type::king) {
        if (c == chess::color::white) { ++white_kings; }
        else { ++black_kings; }
      }
    }

    if (white_kings != 1 || black_kings != 1) {
      return std::nullopt;
    }

    std::int16_t raw_score{0};
    std::int8_t raw_res{0};
    if (!in.read(reinterpret_cast<char*>(&raw_score), sizeof(raw_score))) { return std::nullopt; }
    if (!in.read(reinterpret_cast<char*>(&raw_res), sizeof(raw_res))) { return std::nullopt; }

    result_type res = result_type::undefined;
    switch (raw_res) {
      case 0: res = result_type::loss; break;
      case 1: res = result_type::draw; break;
      case 2: res = result_type::win; break;
      default: res = result_type::undefined; break;
    }

    sample s(board, static_cast<score_type>(raw_score));
    s.set_result(res);
    return s;
  }

  file_reader_iterator<sample> begin() const {
    if (binary_) { return file_reader_iterator<sample>(read_binary, path_, std::ios::binary); }
    return file_reader_iterator<sample>(to_line_reader<sample>(sample::from_string), path_);
  }

  file_reader_iterator<sample> end() const { return file_reader_iterator<sample>(); }

  size_t size() const {
    if (binary_) { return 0; }
    return line_count_size<sample_reader>::size();
  }

  size_t size() {
    if (binary_) { return 0; }
    return line_count_size<sample_reader>::size();
  }

  sample_reader(const std::string& path, const bool binary = false) : path_{path}, binary_{binary} {}
};

}
