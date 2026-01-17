/*
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
 */
#include <eval/nnue.h>

#include <fstream>
#include <iostream>

#include <eval/nnue/io.h>
#include <eval/util/memstream.h>

#define INCBIN_PREFIX g_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <incbin.h>

INCBIN(default_net, EVALFILE);

namespace {
enum class NetworkFlags : eval::u16 {
  None = 0x0000,
};

constexpr eval::u16 ExpectedHeaderVersion = 1;

#pragma pack(push, 1)
struct NetworkHeader {
  std::array<char, 4> magic{};
  eval::u16 version{};
  NetworkFlags flags{};
  eval::u8 padding{};
  eval::u8 arch{};
  eval::u8 activation{};
  eval::u16 hiddenSize{};
  eval::u8 inputBuckets{};
  eval::u8 outputBuckets{};
  eval::u8 nameLen{};
  std::array<char, 48> name{};
};
#pragma pack(pop)

static_assert(sizeof(NetworkHeader) == 64);

inline auto archName(eval::u8 arch) {
  static constexpr auto NetworkArchNames = std::array{"basic", "perspective"};
  if (arch < NetworkArchNames.size()) {
    return NetworkArchNames[arch];
  }
  return "<unknown>";
}

inline auto activationFuncName(eval::u8 func) {
  static constexpr auto ActivationFunctionNames = std::array{"crelu", "screlu", "relu"};
  if (func < ActivationFunctionNames.size()) {
    return ActivationFunctionNames[func];
  }
  return "<unknown>";
}

auto validate(const NetworkHeader &header) {
  if (header.magic != std::array{'C', 'B', 'N', 'F'}) {
    std::cerr << "invalid magic bytes in network header" << std::endl;
    return false;
  }

  if (header.version != ExpectedHeaderVersion) {
    std::cerr << "unsupported network format version " << header.version << " (expected: " << ExpectedHeaderVersion << ")" << std::endl;
    return false;
  }

  if (header.arch != 1) {
    std::cerr << "wrong network architecture " << archName(header.arch) << " (expected: " << archName(1) << ")" << std::endl;
    return false;
  }

  if (header.activation != eval::L1Activation::Id) {
    std::cerr << "wrong network l1 activation function (" << activationFuncName(header.activation)
              << ", expected: " << activationFuncName(eval::L1Activation::Id) << ")" << std::endl;
    return false;
  }

  if (header.hiddenSize != eval::Layer1Size) {
    std::cerr << "wrong number of hidden neurons (" << header.hiddenSize << ", expected: " << eval::Layer1Size << ")" << std::endl;
    return false;
  }

  if (header.inputBuckets != eval::InputFeatureSet::BucketCount) {
    std::cerr << "wrong number of input buckets (" << static_cast<eval::u32>(header.inputBuckets)
              << ", expected: " << eval::InputFeatureSet::BucketCount << ")" << std::endl;
    return false;
  }

  if (header.outputBuckets != eval::OutputBucketing::BucketCount) {
    std::cerr << "wrong number of output buckets (" << static_cast<eval::u32>(header.outputBuckets)
              << ", expected: " << eval::OutputBucketing::BucketCount << ")" << std::endl;
    return false;
  }

  return true;
}

eval::Network s_network{};
std::string s_network_name{};
}  // namespace

namespace eval {
const Network &g_network = s_network;

auto loadDefaultNetwork() -> void {
  const auto *begin = g_default_net_data + sizeof(NetworkHeader);
  const auto *end = g_default_net_data + g_default_net_size;

  const auto *b = reinterpret_cast<const std::byte *>(begin);
  const auto *e = reinterpret_cast<const std::byte *>(end);

  util::MemoryIstream stream{std::span<const std::byte>{b, static_cast<std::size_t>(e - b)}};
  nnue::PaddedParamStream<64> paramStream{stream};
  s_network.readFrom(paramStream);
  s_network_name = std::string(defaultNetworkName());
}

auto loadNetwork(const std::string &name) -> void {
  std::ifstream stream{name, std::ios::binary};
  if (!stream) {
    std::cerr << "failed to open network file \"" << name << "\"" << std::endl;
    return;
  }

  NetworkHeader header{};
  stream.read(reinterpret_cast<char *>(&header), sizeof(NetworkHeader));
  if (!stream) {
    std::cerr << "failed to read network file header" << std::endl;
    return;
  }

  if (!validate(header)) {
    return;
  }

  nnue::PaddedParamStream<64> paramStream{stream};
  if (!s_network.readFrom(paramStream)) {
    std::cerr << "failed to read network parameters" << std::endl;
    return;
  }

  const std::string_view netName{header.name.data(), header.nameLen};
  s_network_name = std::string(netName);
}

auto defaultNetworkName() -> std::string_view {
  const auto &header = *reinterpret_cast<const NetworkHeader *>(g_default_net_data);
  return {header.name.data(), header.nameLen};
}

auto loadedNetworkName() -> std::string_view {
  if (s_network_name.empty()) {
    return "<unknown>";
  }
  return s_network_name;
}
}  // namespace eval
