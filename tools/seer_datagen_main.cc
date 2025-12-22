#include <seer_datagen_main.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#define NNUE_EMBEDDED_WEIGHTS_EXTERN
#include <sample_reader.h>
#include <atomic_data_generator.h>
#undef NNUE_EMBEDDED_WEIGHTS_EXTERN

#include <atomic_sample_writer.h>
#include <search/atomic_tbprobe.h>

namespace {

enum class rescore_mode { search, tb, tb_or_search };


struct options {
  std::string out_path{};
  std::string format{"bin"};
  std::size_t samples{0};
  std::uint64_t seed{1};
  std::size_t threads{1};
  std::size_t progress_every{2000};
  int max_moves{256};
  int eval_limit{6144};
  int fixed_depth{6};
  std::size_t fixed_nodes{5120};
  int plies_min{-1};
  int plies_max{-1};
  std::size_t min_pieces{0};
  double require_capture_prob{0.0};
  std::size_t dedup{0};
  std::size_t dedup_hash_mb{0};
  train::atomic_filter_preset filter{train::atomic_filter_preset::balanced};
  bool allow_mate_in_one{false};
  bool quiet_filter_enabled{true};
  bool use_startpos{false};
  std::vector<std::string> epd_paths{};
  bool has_plies_min{false};
  bool has_plies_max{false};
  bool has_require_capture_prob{false};
  bool has_dedup{false};
  bool has_dedup_hash_mb{false};
};

struct rescore_options {
  std::string in_path{};
  std::string out_path{};
  rescore_mode mode{rescore_mode::search};
  std::string tb_path{};
  std::size_t tb_pieces{6};
  std::size_t threads{1};
  std::size_t nodes{200000};
  int depth{0};
  std::size_t progress_every{2000};
};


void print_usage() {
  std::cout
      << "Usage:\n"
      << "  seer datagen --out PATH --format bin --samples N [options]\n"
      << "  seer datagen gen --out PATH --format bin --samples N [options]\n"
      << "  seer datagen rescore --in PATH --out PATH [options]\n\n"
      << "Gen options:\n"
      << "  --out PATH                 Output .bin path (required).\n"
      << "  --format bin               Output format (only bin supported).\n"
      << "  --samples N                Number of samples to write.\n"
      << "  --seed N                   RNG seed (default: 1).\n"
      << "  --threads N                Concurrency (default: 1).\n"
      << "  --concurrency N            Alias for --threads.\n"
      << "  --progress N              Progress update every N samples (0 = disable, default: 2000).\n"
      << "  --max-moves N              Max plies per game (default: 256).\n"
      << "  --eval-limit N             Score threshold to stop game (default: 6144).\n"
      << "  --fixed-depth N            Max search depth per move (default: 6).\n"
      << "  --fixed-nodes N            Max search nodes per move (default: 5120).\n"
      << "  --plies-min N              Random prelude min plies (default: 8).\n"
      << "  --plies-max N              Random prelude max plies (default: 16).\n"
      << "  --min-pieces N             Minimum total pieces (0 = disabled).\n"
      << "  --require-capture-prob X   Chance to require check/capture (0..1).\n"
      << "  --dedup N                  Dedup window size (records).\n"
      << "  --dedup-hash-mb M           Dedup window size by MB (overrides --dedup).\n"
      << "  --filter minimal|balanced|quiet\n"
      << "  --allow-mate-in-one        Allow mate-in-one positions.\n"
      << "  --no-quiet-filter          Disable quiet-only filter in --filter quiet.\n"
      << "  --startpos                 Include startpos seed.\n"
      << "  --epd PATH                 Load EPD/FEN seeds (repeatable).\n";
}

void print_rescore_usage() {
  std::cout
      << "Usage:\n"
      << "  seer datagen rescore --in PATH --out PATH [options]\n\n"
      << "Rescore options:\n"
      << "  --in PATH                 Input .bin path (required).\n"
      << "  --out PATH                Output .bin path (required).\n"
      << "  --mode search|tb|tb_or_search (default: search).\n"
      << "  --nodes N                 Max nodes per position (default: 200000).\n"
      << "  --depth N                 Max depth per position (0 = disable).\n"
      << "  --threads N               Concurrency (default: 1).\n"
      << "  --progress N              Progress update every N samples (0 = disable, default: 2000).\n"
      << "  --tb-path PATH            Atomic TB directory.\n"
      << "  --tb-pieces N             Max pieces for TB (1..6, default: 6).\n";
}

bool starts_with(const std::string& text, const std::string& prefix) {
  return text.rfind(prefix, 0) == 0;
}

std::string trim(const std::string& text) {
  std::size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) { ++begin; }
  std::size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) { --end; }
  return text.substr(begin, end - begin);
}

std::string strip_underscores(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (const char ch : text) {
    if (ch != '_') { out.push_back(ch); }
  }
  return out;
}

std::optional<std::string> take_value(const std::string& arg, int& idx, const int argc, const char* argv[]) {
  const std::size_t eq = arg.find('=');
  if (eq != std::string::npos) { return arg.substr(eq + 1); }
  if (idx + 1 >= argc) { return std::nullopt; }
  ++idx;
  return std::string(argv[idx]);
}

bool parse_u64(const std::string& text, std::uint64_t& out) {
  const std::string cleaned = strip_underscores(text);
  if (cleaned.empty()) { return false; }
  char* end = nullptr;
  const unsigned long long value = std::strtoull(cleaned.c_str(), &end, 10);
  if (end == cleaned.c_str() || *end != '\0') { return false; }
  out = static_cast<std::uint64_t>(value);
  return true;
}

bool parse_size(const std::string& text, std::size_t& out) {
  std::uint64_t value = 0;
  if (!parse_u64(text, value)) { return false; }
  out = static_cast<std::size_t>(value);
  return true;
}

bool parse_int(const std::string& text, int& out) {
  const std::string cleaned = strip_underscores(text);
  if (cleaned.empty()) { return false; }
  char* end = nullptr;
  const long value = std::strtol(cleaned.c_str(), &end, 10);
  if (end == cleaned.c_str() || *end != '\0') { return false; }
  out = static_cast<int>(value);
  return true;
}

bool parse_double(const std::string& text, double& out) {
  const std::string cleaned = strip_underscores(text);
  if (cleaned.empty()) { return false; }
  char* end = nullptr;
  const double value = std::strtod(cleaned.c_str(), &end);
  if (end == cleaned.c_str() || *end != '\0') { return false; }
  out = value;
  return true;
}

std::optional<train::state_type> parse_fen_relaxed(const std::string& fen) {
  std::istringstream iss(fen);
  std::vector<std::string> tokens{};
  for (std::string tok; iss >> tok;) { tokens.push_back(tok); }
  if (tokens.size() < 4) { return std::nullopt; }
  if (tokens.size() == 4) {
    tokens.push_back("0");
    tokens.push_back("1");
  } else if (tokens.size() == 5) {
    tokens.push_back("1");
  }
  if (tokens.size() < 6) { return std::nullopt; }

  std::ostringstream out;
  for (std::size_t i = 0; i < 6; ++i) {
    if (i) { out << ' '; }
    out << tokens[i];
  }

  try {
    return train::state_type::parse_fen(out.str());
  } catch (...) {
    return std::nullopt;
  }
}

class epd_seed_stream final : public train::seed_provider {
 public:
  explicit epd_seed_stream(const std::vector<std::string>& paths) : paths_(paths) {
    if (paths_.empty()) { current_path_ = 0; }
  }

  bool next(train::state_type& out, std::mt19937_64& /*gen*/) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (paths_.empty()) { return false; }

    const std::size_t start_idx = current_path_;
    bool wrapped = false;

    for (;;) {
      if (!stream_.is_open()) {
        if (!open_current()) { return false; }
      }

      std::string line;
      while (std::getline(stream_, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') { continue; }
        const std::size_t semi = trimmed.find(';');
        const std::string fen = trim(semi == std::string::npos ? trimmed : trimmed.substr(0, semi));
        if (fen.empty()) { continue; }
        const auto parsed = parse_fen_relaxed(fen);
        if (!parsed.has_value()) { continue; }
        out = *parsed;
        return true;
      }

      stream_.close();
      stream_.clear();
      current_path_ = (current_path_ + 1) % paths_.size();
      if (current_path_ == start_idx) {
        if (wrapped) { return false; }
        wrapped = true;
      }
    }
  }

 private:
  bool open_current() {
    if (paths_.empty()) { return false; }
    stream_.close();
    stream_.clear();
    for (std::size_t i = 0; i < paths_.size(); ++i) {
      stream_.open(paths_[current_path_]);
      if (stream_) { return true; }
      stream_.close();
      stream_.clear();
      current_path_ = (current_path_ + 1) % paths_.size();
    }
    return false;
  }

  std::vector<std::string> paths_{};
  std::size_t current_path_{0};
  std::ifstream stream_{};
  std::mutex mutex_{};
};

struct rescore_stream {
  std::mutex mutex_{};
  std::ifstream file_{};
  bool ok_{false};

  explicit rescore_stream(const std::string& path) : file_(path, std::ios::binary), ok_(static_cast<bool>(file_)) {}

  bool ok() const { return ok_; }

  bool next(train::sample& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_) { return false; }
    const auto opt = train::sample_reader::read_binary(file_);
    if (!opt.has_value()) { return false; }
    out = *opt;
    return true;
  }
};

struct rescore_writer {
  std::mutex mutex_{};
  train::atomic_sample_writer writer_;
  std::atomic_uint64_t completed_{0};

  std::uint64_t report_every_{0};
  std::uint64_t next_report_{0};
  std::chrono::steady_clock::time_point start_time_{};
  bool progress_enabled_{false};

  void set_progress_every(const std::uint64_t report_every) {
    std::lock_guard<std::mutex> lock(mutex_);
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

  void write_sample(const train::sample& sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    writer_.append_sample(sample);
    ++completed_;
    const auto completed = completed_.load();
    if (progress_enabled_ && report_every_ > 0 && completed >= next_report_) {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(now - start_time_).count();
      const double rate = elapsed > 0.0 ? static_cast<double>(completed) / elapsed : 0.0;
      const std::uint64_t rate_int = static_cast<std::uint64_t>(rate);
      std::cerr << "progress " << completed << " samples " << rate_int << " samples/s\n";
      std::cerr.flush();
      next_report_ = ((completed / report_every_) + 1) * report_every_;
    }
  }

  explicit rescore_writer(const std::string& path) : writer_(path) {}
};



bool parse_rescore_mode(const std::string& text, rescore_mode& out) {
  if (text == "search") {
    out = rescore_mode::search;
    return true;
  }
  if (text == "tb") {
    out = rescore_mode::tb;
    return true;
  }
  if (text == "tb_or_search") {
    out = rescore_mode::tb_or_search;
    return true;
  }
  return false;
}

bool parse_filter(const std::string& text, train::atomic_filter_preset& out) {
  if (text == "minimal") {
    out = train::atomic_filter_preset::minimal;
    return true;
  }
  if (text == "balanced") {
    out = train::atomic_filter_preset::balanced;
    return true;
  }
  if (text == "quiet") {
    out = train::atomic_filter_preset::quiet;
    return true;
  }
  return false;
}

}  // namespace

int seer_datagen_gen_main(const int argc, const char* argv[]) {
  options opts{};

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    }

    if (starts_with(arg, "--out")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || value->empty()) {
        std::cerr << "error: --out requires a path\n";
        return 2;
      }
      opts.out_path = *value;
      continue;
    }

    if (starts_with(arg, "--format")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || value->empty()) {
        std::cerr << "error: --format requires a value\n";
        return 2;
      }
      opts.format = *value;
      continue;
    }

    if (starts_with(arg, "--samples")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.samples)) {
        std::cerr << "error: --samples expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--seed")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_u64(*value, opts.seed)) {
        std::cerr << "error: --seed expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--threads") || starts_with(arg, "--concurrency")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.threads)) {
        std::cerr << "error: --threads expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--progress")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.progress_every)) {
        std::cerr << "error: --progress expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--max-moves")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_int(*value, opts.max_moves)) {
        std::cerr << "error: --max-moves expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--eval-limit")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_int(*value, opts.eval_limit)) {
        std::cerr << "error: --eval-limit expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--fixed-depth")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_int(*value, opts.fixed_depth)) {
        std::cerr << "error: --fixed-depth expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--fixed-nodes")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.fixed_nodes)) {
        std::cerr << "error: --fixed-nodes expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--plies-min")) {
      const auto value = take_value(arg, i, argc, argv);
      int parsed = 0;
      if (!value.has_value() || !parse_int(*value, parsed)) {
        std::cerr << "error: --plies-min expects an integer\n";
        return 2;
      }
      opts.plies_min = parsed;
      opts.has_plies_min = true;
      continue;
    }

    if (starts_with(arg, "--plies-max")) {
      const auto value = take_value(arg, i, argc, argv);
      int parsed = 0;
      if (!value.has_value() || !parse_int(*value, parsed)) {
        std::cerr << "error: --plies-max expects an integer\n";
        return 2;
      }
      opts.plies_max = parsed;
      opts.has_plies_max = true;
      continue;
    }

    if (starts_with(arg, "--min-pieces")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.min_pieces)) {
        std::cerr << "error: --min-pieces expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--require-capture-prob")) {
      const auto value = take_value(arg, i, argc, argv);
      double parsed = 0.0;
      if (!value.has_value() || !parse_double(*value, parsed)) {
        std::cerr << "error: --require-capture-prob expects a number\n";
        return 2;
      }
      opts.require_capture_prob = parsed;
      opts.has_require_capture_prob = true;
      continue;
    }

    if (starts_with(arg, "--dedup-hash-mb")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.dedup_hash_mb)) {
        std::cerr << "error: --dedup-hash-mb expects an integer\n";
        return 2;
      }
      opts.has_dedup_hash_mb = true;
      continue;
    }

    if (starts_with(arg, "--dedup")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.dedup)) {
        std::cerr << "error: --dedup expects an integer\n";
        return 2;
      }
      opts.has_dedup = true;
      continue;
    }

    if (starts_with(arg, "--filter")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_filter(*value, opts.filter)) {
        std::cerr << "error: --filter must be minimal, balanced, or quiet\n";
        return 2;
      }
      continue;
    }

    if (arg == "--allow-mate-in-one") {
      opts.allow_mate_in_one = true;
      continue;
    }

    if (starts_with(arg, "--allow-mate-in-one")) {
      const auto value = take_value(arg, i, argc, argv);
      int parsed = 0;
      if (!value.has_value() || !parse_int(*value, parsed)) {
        std::cerr << "error: --allow-mate-in-one expects 0 or 1\n";
        return 2;
      }
      opts.allow_mate_in_one = (parsed != 0);
      continue;
    }

    if (arg == "--no-quiet-filter") {
      opts.quiet_filter_enabled = false;
      continue;
    }

    if (arg == "--startpos") {
      opts.use_startpos = true;
      continue;
    }

    if (starts_with(arg, "--epd")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || value->empty()) {
        std::cerr << "error: --epd requires a path\n";
        return 2;
      }
      opts.epd_paths.push_back(*value);
      continue;
    }

    std::cerr << "error: unknown option: " << arg << "\n";
    return 2;
  }

  if (opts.out_path.empty()) {
    std::cerr << "error: --out is required\n";
    print_usage();
    return 2;
  }
  if (opts.samples == 0) {
    std::cerr << "error: --samples must be > 0\n";
    return 2;
  }
  if (opts.format != "bin") {
    std::cerr << "error: only --format bin is supported\n";
    return 2;
  }
  if (opts.max_moves <= 0) {
    std::cerr << "error: --max-moves must be > 0\n";
    return 2;
  }
  if (opts.fixed_depth <= 0) {
    std::cerr << "error: --fixed-depth must be > 0\n";
    return 2;
  }
  if (opts.fixed_nodes == 0) {
    std::cerr << "error: --fixed-nodes must be > 0\n";
    return 2;
  }
  if (opts.threads == 0) { opts.threads = 1; }

  if (!opts.has_plies_min && !opts.has_plies_max) {
    opts.plies_min = 8;
    opts.plies_max = 16;
  } else if (!opts.has_plies_min) {
    opts.plies_min = opts.plies_max;
  } else if (!opts.has_plies_max) {
    opts.plies_max = opts.plies_min;
  }

  if (opts.plies_min < 0 || opts.plies_max < 0) {
    std::cerr << "error: --plies-min/max must be >= 0\n";
    return 2;
  }

  if (opts.require_capture_prob < 0.0 || opts.require_capture_prob > 1.0) {
    std::cerr << "error: --require-capture-prob must be between 0 and 1\n";
    return 2;
  }

  constexpr std::size_t default_dedup = 1000000;
  constexpr double default_require_capture_prob = 0.2;
  if (!opts.has_require_capture_prob) {
    if (opts.filter == train::atomic_filter_preset::balanced) {
      opts.require_capture_prob = default_require_capture_prob;
    } else {
      opts.require_capture_prob = 0.0;
    }
  }

  if (!opts.has_dedup && !opts.has_dedup_hash_mb) {
    if (opts.filter == train::atomic_filter_preset::balanced || opts.filter == train::atomic_filter_preset::quiet) {
      opts.dedup = default_dedup;
    } else {
      opts.dedup = 0;
    }
  }

  std::size_t dedup_capacity = opts.dedup;
  if (opts.has_dedup_hash_mb) {
    const std::size_t bytes = opts.dedup_hash_mb * 1024ULL * 1024ULL;
    dedup_capacity = bytes / sizeof(std::uint64_t);
  }

  std::vector<train::state_type> seeds{};
  if (opts.use_startpos || (opts.epd_paths.empty() && !opts.use_startpos)) {
    seeds.push_back(train::state_type::start_pos());
  }

  std::shared_ptr<train::seed_provider> seed_provider{};
  if (!opts.epd_paths.empty()) {
    for (const auto& path : opts.epd_paths) {
      std::ifstream in(path);
      if (!in) {
        std::cerr << "error: unable to open epd file: " << path << "\n";
        return 2;
      }
    }
    seed_provider = std::make_shared<epd_seed_stream>(opts.epd_paths);
  }

  constexpr std::size_t default_tt_mb = 128;

  train::atomic_data_generator gen(opts.out_path, opts.samples, default_tt_mb);
  gen.set_concurrency(opts.threads)
      .set_ply_limit(static_cast<search::depth_type>(opts.max_moves))
      .set_eval_limit(static_cast<search::score_type>(opts.eval_limit))
      .set_fixed_depth(static_cast<search::depth_type>(opts.fixed_depth))
      .set_fixed_nodes(opts.fixed_nodes)
      .set_random_ply_range(static_cast<search::depth_type>(opts.plies_min), static_cast<search::depth_type>(opts.plies_max))
      .set_min_pieces(opts.min_pieces)
      .set_require_capture_prob(opts.require_capture_prob)
      .set_filter(opts.filter)
      .set_quiet_filter_enabled(opts.quiet_filter_enabled)
      .set_allow_mate_in_one(opts.allow_mate_in_one)
      .set_seed(opts.seed)
      .set_seed_provider(seed_provider)
      .set_progress_every(opts.progress_every)
      .set_dedup_capacity(dedup_capacity)
      .set_seeds(seeds);

  gen.generate_data();
  return 0;
}

int seer_datagen_rescore_main(const int argc, const char* argv[]) {
  rescore_options opts{};

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_rescore_usage();
      return 0;
    }

    if (starts_with(arg, "--in")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || value->empty()) {
        std::cerr << "error: --in requires a path\n";
        return 2;
      }
      opts.in_path = *value;
      continue;
    }

    if (starts_with(arg, "--out")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || value->empty()) {
        std::cerr << "error: --out requires a path\n";
        return 2;
      }
      opts.out_path = *value;
      continue;
    }

    if (starts_with(arg, "--mode")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_rescore_mode(*value, opts.mode)) {
        std::cerr << "error: --mode must be search, tb, or tb_or_search\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--nodes")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.nodes)) {
        std::cerr << "error: --nodes expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--depth")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_int(*value, opts.depth)) {
        std::cerr << "error: --depth expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--threads") || starts_with(arg, "--concurrency")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.threads)) {
        std::cerr << "error: --threads expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--progress")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.progress_every)) {
        std::cerr << "error: --progress expects an integer\n";
        return 2;
      }
      continue;
    }

    if (starts_with(arg, "--tb-path")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || value->empty()) {
        std::cerr << "error: --tb-path requires a path\n";
        return 2;
      }
      opts.tb_path = *value;
      continue;
    }

    if (starts_with(arg, "--tb-pieces")) {
      const auto value = take_value(arg, i, argc, argv);
      if (!value.has_value() || !parse_size(*value, opts.tb_pieces)) {
        std::cerr << "error: --tb-pieces expects an integer\n";
        return 2;
      }
      continue;
    }

    std::cerr << "error: unknown option: " << arg << "\n";
    return 2;
  }

  if (opts.in_path.empty()) {
    std::cerr << "error: --in is required\n";
    print_rescore_usage();
    return 2;
  }
  if (opts.out_path.empty()) {
    std::cerr << "error: --out is required\n";
    print_rescore_usage();
    return 2;
  }
  if (opts.nodes == 0 && opts.depth <= 0) {
    std::cerr << "error: --nodes or --depth must be set\n";
    return 2;
  }
  if (opts.threads == 0) { opts.threads = 1; }
  if (opts.tb_pieces == 0 || opts.tb_pieces > 6) {
    std::cerr << "error: --tb-pieces must be between 1 and 6\n";
    return 2;
  }

  rescore_stream stream(opts.in_path);
  if (!stream.ok()) {
    std::cerr << "error: unable to open input bin: " << opts.in_path << "\n";
    return 2;
  }

  rescore_writer writer(opts.out_path);
  writer.set_progress_every(opts.progress_every);

  const nnue::quantized_weights q_weights = [&] {
    nnue::weights tmp{};
    nnue::embedded_weight_streamer embedded(nnue::embed::weights_file_data);
    tmp.load(embedded);
    return tmp.to<nnue::quantized_weights>();
  }();

  auto constants = std::make_shared<search::search_constants>(opts.threads);
  auto tt = std::make_shared<search::transposition_table>(128);

  const bool allow_search = (opts.mode == rescore_mode::search || opts.mode == rescore_mode::tb_or_search);
  bool want_tb = (opts.mode != rescore_mode::search);
  bool tb_ready = false;
  if (want_tb) {
    if (opts.tb_path.empty()) {
      if (opts.mode == rescore_mode::tb) {
        std::cerr << "error: --tb-path is required for --mode tb\n";
        return 2;
      }
      want_tb = false;
    } else {
      tb_ready = search::atomic_tb::init(opts.tb_path);
      if (!tb_ready) {
        if (opts.mode == rescore_mode::tb) {
          std::cerr << "error: unable to init TB path: " << opts.tb_path << "\n";
          return 2;
        }
        std::cerr << "warning: unable to init TB path: " << opts.tb_path << "; using search only\n";
        want_tb = false;
      }
    }
  }

  std::mutex tb_mutex{};
  auto apply_tb = [&](train::sample& sample) -> bool {
    if (!want_tb) { return false; }
    if (sample.state_.num_pieces() > opts.tb_pieces) { return false; }
    search::atomic_tb::ProbeResult probe{};
    {
      std::lock_guard<std::mutex> lock(tb_mutex);
      if (!search::atomic_tb::probe_wdl(sample.state_, probe)) { return false; }
    }
    switch (probe.wdl) {
      case search::atomic_tb::WDL::Win:
        sample.result_ = train::result_type::win;
        sample.score_ = search::tb_win_score;
        break;
      case search::atomic_tb::WDL::Loss:
        sample.result_ = train::result_type::loss;
        sample.score_ = search::tb_loss_score;
        break;
      case search::atomic_tb::WDL::Draw:
        sample.result_ = train::result_type::draw;
        sample.score_ = search::draw_score;
        break;
      default:
        return false;
    }
    return true;
  };

  auto worker_fn = [&](const std::size_t thread_idx) {
    (void)thread_idx;
    using worker_type = search::search_worker;
    std::unique_ptr<worker_type> worker = std::make_unique<worker_type>(
        &q_weights, tt, constants,
        [&](const auto& w) { if (opts.depth > 0 && w.depth() >= opts.depth) { worker->stop(); } },
        [&](const auto& w) { if (opts.nodes > 0 && w.nodes() >= opts.nodes) { worker->stop(); } });

    train::sample sample{};
    while (stream.next(sample)) {
      (worker->internal).reset();
      const bool tb_used = apply_tb(sample);
      if (!tb_used && allow_search) {
        chess::board_history hist{};
        worker->go(hist, sample.state_, 1);
        worker->iterative_deepening_loop();
        worker->stop();
        sample.score_ = worker->score();
      }
      writer.write_sample(sample);
    }
  };

  std::vector<std::thread> threads{};
  for (std::size_t i = 0; i < opts.threads; ++i) { threads.emplace_back(worker_fn, i); }
  for (auto& thread : threads) { thread.join(); }
  if (tb_ready) { search::atomic_tb::close(); }
  return 0;
}

int seer_datagen_main(const int argc, const char* argv[]) {
  if (argc >= 2) {
    const std::string sub = argv[1];
    if (sub == "rescore") { return seer_datagen_rescore_main(argc - 1, argv + 1); }
    if (sub == "gen") { return seer_datagen_gen_main(argc - 1, argv + 1); }
  }
  return seer_datagen_gen_main(argc, argv);
}
