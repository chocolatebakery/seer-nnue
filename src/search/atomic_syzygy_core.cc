/*
  Atomic Syzygy core loader/decoder (WDL-only scaffold)
  Commit A: load .atbw files (validate magic) into memory.
  Probing still stubbed; indexing/decoding will follow.
*/

#include <search/atomic_syzygy_core.h>

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <unordered_map>
#include <iostream>
#include <vector>

#include <search/atomic_tbprobe.h>
#include <chess/board.h>

namespace {

// ------------------------
// Low-level helpers (LE)
// ------------------------
[[maybe_unused]] inline uint16_t read_le_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

[[maybe_unused]] inline uint32_t read_le_u32(const uint8_t* p) {
  return p[0] | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

[[maybe_unused]] inline uint32_t read_be_u32(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

[[maybe_unused]] inline uint64_t read_be_u64(const uint8_t* p) {
  return (static_cast<uint64_t>(p[0]) << 56) | (static_cast<uint64_t>(p[1]) << 48) |
         (static_cast<uint64_t>(p[2]) << 40) | (static_cast<uint64_t>(p[3]) << 32) |
         (static_cast<uint64_t>(p[4]) << 24) | (static_cast<uint64_t>(p[5]) << 16) |
         (static_cast<uint64_t>(p[6]) << 8) | static_cast<uint64_t>(p[7]);
}

// ------------------------
// Pairs (Huffman) skeleton
// ------------------------
// This mirrors the Fathom-style PairsData layout used by Syzygy tables.
// Decode plumbing will be wired later; for now we keep the structure and
// lifecycle hooks so we can plug in setup_pairs()/decompress_pairs in the
// next step without churning the loader API again.
struct PairsData {
  uint8_t* indexTable{nullptr};
  uint16_t* sizeTable{nullptr};
  uint8_t* data{nullptr};
  uint16_t* offset{nullptr};
  uint8_t* symLen{nullptr};
  uint8_t* symPat{nullptr};
  uint32_t numSyms{0};
  uint8_t blockSize{0};
  uint8_t idxBits{0};
  uint8_t minLen{0};
  uint8_t constValue[2]{0, 0};
  uint64_t base[1]{0};

  PairsData() = default;
};

struct LoadedTable {
  std::string filename;
  std::string stem;
  std::vector<uint8_t> data;
  bool ok{false};
  // WDL decoding metadata (to be populated once pairs/indexing are wired)
  PairsData* wdl_pairs[2]{nullptr, nullptr};
  uint64_t tb_size[2]{0, 0};
  bool wdl_ready{false};
  bool has_pawns{false};
  uint8_t wpawns{0};
  uint8_t bpawns{0};
  uint8_t num_pieces{0};
  bool kk_enc{false};
  bool split{false};
};

struct MaterialSig {
  uint8_t cnt[12]{};

  bool operator==(const MaterialSig& o) const noexcept {
    for (int i = 0; i < 12; ++i) {
      if (cnt[i] != o.cnt[i]) { return false; }
    }
    return true;
  }
};

struct MaterialSigHash {
  std::size_t operator()(const MaterialSig& k) const noexcept {
    std::size_t h = 0;
    for (int i = 0; i < 12; ++i) { h = h * static_cast<std::size_t>(131) + k.cnt[i]; }
    return h;
  }
};

std::vector<LoadedTable> g_tables;
std::unordered_map<MaterialSig, std::size_t, MaterialSigHash> g_key_to_table;
std::string g_tb_path;
bool g_inited = false;

// Atomic WDL magics (from Multivariant/Stockfish atomic tbprobe)
constexpr uint8_t ATOMIC_WDL_MAGIC[2][4] = {
    {0x91, 0xA9, 0x5E, 0xEB},
    {0x55, 0x8D, 0xA4, 0x49},
};

bool read_all_bytes(const std::filesystem::path& p, std::vector<uint8_t>& out) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { return false; }
  f.seekg(0, std::ios::end);
  const std::streamsize sz = f.tellg();
  if (sz <= 0) { return false; }
  f.seekg(0, std::ios::beg);
  out.resize(static_cast<std::size_t>(sz));
  return static_cast<bool>(f.read(reinterpret_cast<char*>(out.data()), sz));
}

bool has_atomic_magic(const std::vector<uint8_t>& data) {
  if (data.size() < 4) { return false; }
  for (const auto& m : ATOMIC_WDL_MAGIC) {
    if (data[0] == m[0] && data[1] == m[1] && data[2] == m[2] && data[3] == m[3]) { return true; }
  }
  return false;
}

// ------------------------
// Pairs decode (Fathom-style)
// ------------------------

static void calc_symLen(PairsData* d, uint32_t s, uint8_t* tmp) {
  if (s >= d->numSyms) { return; }
  if (tmp[s]) { return; }  // already visited
  tmp[s] = 1;

  uint8_t* w = d->symPat + 3 * s;
  const uint32_t s2 = (static_cast<uint32_t>(w[2]) << 4) | (w[1] >> 4);
  if (s2 == 0x0fff) {
    d->symLen[s] = 0;
    return;
  }

  const uint32_t s1 = ((w[1] & 0x0f) << 8) | w[0];
  if (s1 >= d->numSyms || s2 >= d->numSyms) { return; }

  calc_symLen(d, s1, tmp);
  calc_symLen(d, s2, tmp);
  d->symLen[s] = static_cast<uint8_t>(d->symLen[s1] + d->symLen[s2] + 1);
}

// Setup pairs block; size[] is populated with section sizes (index, sizeTable, data).
static PairsData* setup_pairs(uint8_t** ptr, const uint8_t* end, uint64_t tb_size, size_t size[3], uint8_t& flags) {
  uint8_t* data = *ptr;
  if (data >= end) { return nullptr; }
  flags = data[0];

  if (flags & 0x80) {
    auto* d = static_cast<PairsData*>(std::calloc(1, sizeof(PairsData)));
    if (!d) { return nullptr; }
    d->idxBits = 0;
    d->constValue[0] = data[1];
    d->constValue[1] = 0;
    *ptr = data + 2;
    size[0] = size[1] = size[2] = 0;
    return d;
  }

  const uint8_t blockSize = data[1];
  const uint8_t idxBits = data[2];
  const uint32_t realNumBlocks = read_le_u32(data + 4);
  const uint32_t numBlocks = realNumBlocks + data[3];
  const int maxLen = data[8];
  const int minLen = data[9];
  const int h = maxLen - minLen + 1;
  const uint32_t numSyms = read_le_u16(data + 10 + 2 * h);

  const size_t header_need = 12 + 2 * h + 3 * static_cast<size_t>(numSyms) + (numSyms & 1);
  if (data + header_need > end) { return nullptr; }

  auto* d = static_cast<PairsData*>(std::calloc(1, sizeof(PairsData) + h * sizeof(uint64_t) + numSyms));
  if (!d) { return nullptr; }
  d->blockSize = blockSize;
  d->idxBits = idxBits;
  d->numSyms = numSyms;
  d->offset = reinterpret_cast<uint16_t*>(&data[10]);
  d->symLen = reinterpret_cast<uint8_t*>(d) + sizeof(PairsData) + h * sizeof(uint64_t);
  d->symPat = &data[12 + 2 * h];
  d->minLen = static_cast<uint8_t>(minLen);
  *ptr = &data[12 + 2 * h + 3 * numSyms + (numSyms & 1)];

  const size_t num_indices = (tb_size + (static_cast<uint64_t>(1) << idxBits) - 1) >> idxBits;
  size[0] = 6ULL * num_indices;
  size[1] = 2ULL * numBlocks;
  size[2] = static_cast<size_t>(realNumBlocks) << blockSize;

  // Compute symLen via recursion
  if (numSyms == 0 || numSyms > 4096) {
    std::free(d);
    return nullptr;
  }
  std::vector<uint8_t> tmp(numSyms, 0);
  for (uint32_t s = 0; s < numSyms; ++s) {
    if (!tmp[s]) { calc_symLen(d, s, tmp.data()); }
  }

  // base[] reconstruction
  d->base[h - 1] = 0;
  for (int i = h - 2; i >= 0; --i) {
    const uint64_t up = static_cast<uint64_t>(read_le_u16(reinterpret_cast<uint8_t*>(d->offset + i)));
    const uint64_t dn = static_cast<uint64_t>(read_le_u16(reinterpret_cast<uint8_t*>(d->offset + i + 1)));
    d->base[i] = (d->base[i + 1] + up - dn) / 2;
  }
  for (int i = 0; i < h; ++i) { d->base[i] <<= (64 - (minLen + i)); }

  d->offset -= d->minLen;  // allow offset[l] with l >= minLen
  return d;
}

// Decompress a single symbol; returns pointer into symPat (3-byte node) or constValue
// static const uint8_t* decompress_pairs(const PairsData* d, size_t idx) {
//   if (!d) { return nullptr; }
//   if (!d->idxBits) { return d->constValue; }
//
//   uint32_t mainIdx = static_cast<uint32_t>(idx >> d->idxBits);
//   int litIdx = static_cast<int>(idx & (((size_t)1 << d->idxBits) - 1)) - (static_cast<size_t>(1) << (d->idxBits - 1));
//   uint32_t block = read_le_u32(d->indexTable + 6 * mainIdx);
//   const uint16_t idxOffset = read_le_u16(reinterpret_cast<const uint8_t*>(d->indexTable + 6 * mainIdx + 4));
//   litIdx += static_cast<int>(idxOffset);
//
//   if (litIdx < 0) {
//     while (litIdx < 0) { litIdx += d->sizeTable[--block] + 1; }
//   } else {
//     while (litIdx > d->sizeTable[block]) { litIdx -= d->sizeTable[block++] + 1; }
//   }
//
//   const uint8_t* ptr = d->data + (static_cast<size_t>(block) << d->blockSize);
//
//   const int m = d->minLen;
//   const uint16_t* offset = d->offset;
//   const uint64_t* base = d->base - m;
//   const uint8_t* symLen = d->symLen;
//   uint32_t sym;
//   uint32_t bitCnt = 0;
//
//   uint64_t code = read_be_u64(ptr);
//   ptr += 8;
//   for (;;) {
//     int l = m;
//     while (code < base[l]) { ++l; }
//     sym = offset[l];
//     sym += static_cast<uint32_t>((code - base[l]) >> (64 - l));
//     if (litIdx < static_cast<int>(symLen[sym]) + 1) { break; }
//     litIdx -= static_cast<int>(symLen[sym]) + 1;
//     code <<= l;
//     bitCnt += l;
//     if (bitCnt >= 32) {
//       bitCnt -= 32;
//       const uint32_t tmp = read_be_u32(ptr);
//       ptr += 4;
//       code |= static_cast<uint64_t>(tmp) << bitCnt;
//     }
//   }
//
//   const uint8_t* symPat = d->symPat;
//   while (symLen[sym] != 0) {
//     const uint8_t* w = symPat + (3 * sym);
//     const int s1 = ((w[1] & 0x0f) << 8) | w[0];
//     if (litIdx < static_cast<int>(symLen[s1]) + 1) {
//       sym = s1;
//     } else {
//       litIdx -= static_cast<int>(symLen[s1]) + 1;
//       sym = (w[2] << 4) | (w[1] >> 4);
//     }
//   }
//
//   return &symPat[3 * sym];
// }

enum TBPiece : uint8_t { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK };

// Encoding helpers (currently pawnless path only)
constexpr int TB_PIECES = 6;
enum { WDL = 0, PIECE_ENC = 0, FILE_ENC = 1, RANK_ENC = 2 };

struct EncInfo {
  PairsData* precomp{nullptr};
  size_t factor[TB_PIECES]{};
  uint8_t pieces[TB_PIECES]{};
  uint8_t norm[TB_PIECES]{};
};

// Binomial[k][n] for small k<=6, n<=63
size_t Binomial[7][64];

void init_binomial() {
  static bool inited = false;
  if (inited) { return; }
  inited = true;
  for (int k = 0; k < 7; ++k) {
    for (int n = 0; n < 64; ++n) {
      size_t f = 1;
      size_t l = 1;
      for (int i = 0; i < k; ++i) {
        f *= static_cast<size_t>(n - i);
        l *= static_cast<size_t>(i + 1);
      }
      Binomial[k][n] = k == 0 ? 1 : f / l;
    }
  }
}

// Count number of placements of k like pieces on n squares
size_t subfactor(const size_t k, const size_t n) {
  if (k == 0) { return 1; }
  size_t f = n;
  size_t l = 1;
  for (size_t i = 1; i < k; ++i) {
    f *= n - i;
    l *= i + 1;
  }
  return f / l;
}

// init_enc_info adapted (pawnless / PIECE_ENC path only for now)
size_t init_enc_info(EncInfo& ei, const LoadedTable& tbl, const uint8_t* tb, const int shift, const int /*t*/, const int enc) {
  const bool morePawns = enc != PIECE_ENC && tbl.bpawns > 0;
  for (uint8_t i = 0; i < tbl.num_pieces; ++i) {
    ei.pieces[i] = static_cast<uint8_t>((tb[i + 1 + (morePawns ? 1 : 0)] >> shift) & 0x0f);
    ei.norm[i] = 0;
  }

  const int order = (tb[0] >> shift) & 0x0f;
  const int order2 = morePawns ? (tb[1] >> shift) & 0x0f : 0x0f;

  int k = 0;
  if (enc != PIECE_ENC) {
    // pawns not handled yet
    return 0;
  } else {
    k = tbl.kk_enc ? 2 : 3;
    ei.norm[0] = static_cast<uint8_t>(k);
  }

  if (morePawns) {
    ei.norm[k] = tbl.bpawns;
    k += ei.norm[k];
  }

  for (int i = k; i < tbl.num_pieces; i += ei.norm[i]) {
    for (int j = i; j < tbl.num_pieces && ei.pieces[j] == ei.pieces[i]; ++j) {
      ei.norm[i]++;
    }
  }

  int n = 64 - k;
  size_t f = 1;

  for (int i = 0; k < tbl.num_pieces || i == order || i == order2; ++i) {
    if (i == order) {
      ei.factor[0] = f;
      f *= tbl.kk_enc ? 462 : 31332;
    } else if (i == order2) {
      ei.factor[ei.norm[0]] = f;
      f *= subfactor(ei.norm[ei.norm[0]], 48 - ei.norm[0]);
    } else {
      ei.factor[k] = f;
      f *= subfactor(ei.norm[k], n);
      n -= ei.norm[k];
      k += ei.norm[k];
    }
  }
  return f;
}

// int num_tables(const LoadedTable& tbl) { return tbl.has_pawns ? 4 : 1; }

bool parse_material_from_name(const std::string& stem, MaterialSig& out) {
  const auto pos = stem.find('v');
  if (pos == std::string::npos) { return false; }

  auto parse_side = [&](std::string_view s, const bool white) -> bool {
    for (const char c : s) {
      switch (c) {
        case 'K': out.cnt[white ? WK : BK]++; break;
        case 'Q': out.cnt[white ? WQ : BQ]++; break;
        case 'R': out.cnt[white ? WR : BR]++; break;
        case 'B': out.cnt[white ? WB : BB]++; break;
        case 'N': out.cnt[white ? WN : BN]++; break;
        case 'P': out.cnt[white ? WP : BP]++; break;
        default: return false;
      }
    }
    return true;
  };

  return parse_side(std::string_view(stem).substr(0, pos), true) && parse_side(std::string_view(stem).substr(pos + 1), false);
}

MaterialSig material_from_board(const chess::board& bd) {
  MaterialSig s;
  auto add = [&](chess::square_set bb, TBPiece pc) {
    auto mask = bb.data;
    while (mask) {
      mask &= (mask - 1);
      ++s.cnt[pc];
    }
  };

  add(bd.man_.white.pawn(), TBPiece::WP);
  add(bd.man_.white.knight(), TBPiece::WN);
  add(bd.man_.white.bishop(), TBPiece::WB);
  add(bd.man_.white.rook(), TBPiece::WR);
  add(bd.man_.white.queen(), TBPiece::WQ);
  add(bd.man_.white.king(), TBPiece::WK);

  add(bd.man_.black.pawn(), TBPiece::BP);
  add(bd.man_.black.knight(), TBPiece::BN);
  add(bd.man_.black.bishop(), TBPiece::BB);
  add(bd.man_.black.rook(), TBPiece::BR);
  add(bd.man_.black.queen(), TBPiece::BQ);
  add(bd.man_.black.king(), TBPiece::BK);

  return s;
}

}  // namespace

namespace search::atomic_syzygy_core {

bool init(const std::string& path) noexcept {
  close();
  g_tb_path = path;
  std::error_code ec;
  const std::filesystem::path dir(path);
  if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) { return false; }

  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) { break; }
    if (!entry.is_regular_file()) { continue; }
    const auto& p = entry.path();
    if (p.extension() != ".atbw") { continue; }

    LoadedTable tbl;
    tbl.filename = p.filename().string();
    tbl.stem = p.stem().string();
    if (!read_all_bytes(p, tbl.data)) { continue; }
    tbl.ok = has_atomic_magic(tbl.data);
    g_tables.push_back(std::move(tbl));
  }

  // Build material signature map and try to initialise WDL pairs (pawnless only for now).
  init_binomial();
  g_key_to_table.clear();
  for (std::size_t i = 0; i < g_tables.size(); ++i) {
    auto& tbl = g_tables[i];
    if (!tbl.ok) { continue; }

    MaterialSig sig;
    if (!parse_material_from_name(tbl.stem, sig)) { continue; }
    tbl.has_pawns = (sig.cnt[WP] + sig.cnt[BP]) > 0;
    tbl.wpawns = sig.cnt[WP];
    tbl.bpawns = sig.cnt[BP];
    tbl.num_pieces = 0;
    int singletons = 0;
    for (int c = 0; c < 12; ++c) {
      tbl.num_pieces = static_cast<uint8_t>(tbl.num_pieces + sig.cnt[c]);
      if (sig.cnt[c] == 1) { ++singletons; }
    }
    if (!tbl.has_pawns) { tbl.kk_enc = (singletons == 2); }

    g_key_to_table[sig] = i;

    // For now, skip pawnful tables until pawn encoding is added.
    if (tbl.has_pawns) { continue; }

    // Minimal WDL init: parse header, compute tb_size via init_enc_info, setup pairs and section pointers.
    if (tbl.data.size() < 5) { continue; }
    uint8_t* base = tbl.data.data();
    tbl.split = (base[4] & 0x01) != 0;
    uint8_t* ptr = base + 5;

    EncInfo ei[2]{};
    size_t tb_sizes[2]{};
    size_t sizes[2][3]{};

    tb_sizes[0] = init_enc_info(ei[0], tbl, ptr, 0, 0, PIECE_ENC);
    ptr += tbl.num_pieces + 1;

    if (tbl.split) {
      tb_sizes[1] = init_enc_info(ei[1], tbl, ptr, 4, 0, PIECE_ENC);
      ptr += tbl.num_pieces + 1;
    }

    ptr += reinterpret_cast<uintptr_t>(ptr) & 1;  // align to 2

    std::cerr << "[TB init] " << tbl.stem << " split=" << tbl.split
              << " tb_size0=" << tb_sizes[0] << " tb_size1=" << tb_sizes[1] << std::endl;

    uint8_t flags = 0;
    tbl.wdl_pairs[0] = setup_pairs(&ptr, tbl.data.data() + tbl.data.size(), tb_sizes[0], sizes[0], flags);
    if (!tbl.wdl_pairs[0]) {
      std::cerr << "[TB init] setup_pairs part0 failed for " << tbl.stem << std::endl;
      continue;
    }

    if (tbl.split) {
      // Consume the split table even if unused (to keep ptr advancement correct).
      tbl.wdl_pairs[1] = setup_pairs(&ptr, tbl.data.data() + tbl.data.size(), tb_sizes[1], sizes[1], flags);
      if (!tbl.wdl_pairs[1]) {
        std::cerr << "[TB init] setup_pairs part1 failed for " << tbl.stem << std::endl;
      }
    }

    // indexTable
    tbl.wdl_pairs[0]->indexTable = ptr;
    ptr += sizes[0][0];
    // sizeTable
    tbl.wdl_pairs[0]->sizeTable = reinterpret_cast<uint16_t*>(ptr);
    ptr += sizes[0][1];
    // align to 64-byte boundary for data block
    ptr = reinterpret_cast<uint8_t*>(((reinterpret_cast<uintptr_t>(ptr) + 0x3fu) & ~static_cast<uintptr_t>(0x3f)));
    tbl.wdl_pairs[0]->data = ptr;
    ptr += sizes[0][2];

    if (tbl.split && tbl.wdl_pairs[1]) {
      tbl.wdl_pairs[1]->indexTable = ptr;
      ptr += sizes[1][0];
      tbl.wdl_pairs[1]->sizeTable = reinterpret_cast<uint16_t*>(ptr);
      ptr += sizes[1][1];
      ptr = reinterpret_cast<uint8_t*>(((reinterpret_cast<uintptr_t>(ptr) + 0x3fu) & ~static_cast<uintptr_t>(0x3f)));
      tbl.wdl_pairs[1]->data = ptr;
      ptr += sizes[1][2];
    }

    tbl.tb_size[0] = tb_sizes[0];
    tbl.tb_size[1] = tbl.split ? tb_sizes[1] : 0;
    std::cerr << "[TB init] ready " << tbl.stem << " split=" << tbl.split
              << " tb_size0=" << tbl.tb_size[0] << " tb_size1=" << tbl.tb_size[1] << std::endl;
    tbl.wdl_ready = true;
  }

  g_inited = true;
  // Return true if at least one valid table was loaded and indexed
  for (const auto& t : g_tables) {
    if (t.ok) { return true; }
  }
  return false;
}

bool probe_wdl(const chess::board& bd, atomic_tb::ProbeResult& out) noexcept {
  if (!g_inited) { return false; }
  MaterialSig sig = material_from_board(bd);
  const auto it = g_key_to_table.find(sig);
  if (it == g_key_to_table.end()) {
    return false;
  }
  const auto& tbl = g_tables[it->second];
  if (!tbl.wdl_ready || !tbl.wdl_pairs[0]) { return false; }

  // TODO: implementar encode + decode real e devolver WDL
  (void)out;
  return false;
}

void close() noexcept {
  for (auto& t : g_tables) {
    for (auto* p : t.wdl_pairs) { free(p); }
    t.wdl_pairs[0] = t.wdl_pairs[1] = nullptr;
  }
  g_tables.clear();
  g_key_to_table.clear();
  g_tb_path.clear();
  g_inited = false;
}

}  // namespace search::atomic_syzygy_core
