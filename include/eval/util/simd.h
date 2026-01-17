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
#pragma once

#include "../../eval/types.h"

#include <algorithm>
#include <cassert>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE4_1__)
#include <immintrin.h>
#endif

namespace eval::util::simd {
#if defined(__AVX512F__)
using VectorI16 = __m512i;
using VectorI32 = __m512i;
constexpr std::uintptr_t Alignment = sizeof(VectorI16);
#elif defined(__AVX2__)
using VectorI16 = __m256i;
using VectorI32 = __m256i;
constexpr std::uintptr_t Alignment = sizeof(VectorI16);
#elif defined(__SSE4_1__)
using VectorI16 = __m128i;
using VectorI32 = __m128i;
constexpr std::uintptr_t Alignment = sizeof(VectorI16);
#else
using VectorI16 = i16;
using VectorI32 = i32;
constexpr std::uintptr_t Alignment = 16;
#endif

constexpr usize ChunkSize = sizeof(VectorI16) / sizeof(i16);

#define EVAL_SIMD_ALIGNAS alignas(eval::util::simd::Alignment)

template <std::uintptr_t A = Alignment, typename T = void>
inline auto isAligned(const T *ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) % A) == 0;
}

namespace impl {
EVAL_ALWAYS_INLINE_NDEBUG inline auto zeroI16() -> VectorI16 {
#if defined(__AVX512F__)
  return _mm512_setzero_si512();
#elif defined(__AVX2__)
  return _mm256_setzero_si256();
#elif defined(__SSE4_1__)
  return _mm_setzero_si128();
#else
  return 0;
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto set1I16(i16 v) -> VectorI16 {
#if defined(__AVX512F__)
  return _mm512_set1_epi16(v);
#elif defined(__AVX2__)
  return _mm256_set1_epi16(v);
#elif defined(__SSE4_1__)
  return _mm_set1_epi16(v);
#else
  return v;
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto loadI16(const void *ptr) -> VectorI16 {
  assert(isAligned(ptr));
#if defined(__AVX512F__)
  return _mm512_load_si512(ptr);
#elif defined(__AVX2__)
  return _mm256_load_si256(static_cast<const VectorI16 *>(ptr));
#elif defined(__SSE4_1__)
  return _mm_load_si128(static_cast<const VectorI16 *>(ptr));
#else
  return *static_cast<const VectorI16 *>(ptr);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto storeI16(void *ptr, VectorI16 v) -> void {
  assert(isAligned(ptr));
#if defined(__AVX512F__)
  _mm512_store_si512(ptr, v);
#elif defined(__AVX2__)
  _mm256_store_si256(static_cast<VectorI16 *>(ptr), v);
#elif defined(__SSE4_1__)
  _mm_store_si128(static_cast<VectorI16 *>(ptr), v);
#else
  *static_cast<VectorI16 *>(ptr) = v;
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto minI16(VectorI16 a, VectorI16 b) -> VectorI16 {
#if defined(__AVX512F__)
  return _mm512_min_epi16(a, b);
#elif defined(__AVX2__)
  return _mm256_min_epi16(a, b);
#elif defined(__SSE4_1__)
  return _mm_min_epi16(a, b);
#else
  return std::min(a, b);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto maxI16(VectorI16 a, VectorI16 b) -> VectorI16 {
#if defined(__AVX512F__)
  return _mm512_max_epi16(a, b);
#elif defined(__AVX2__)
  return _mm256_max_epi16(a, b);
#elif defined(__SSE4_1__)
  return _mm_max_epi16(a, b);
#else
  return std::max(a, b);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto clampI16(VectorI16 v, VectorI16 lo, VectorI16 hi) -> VectorI16 {
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE4_1__)
  return minI16(maxI16(v, lo), hi);
#else
  return std::clamp(v, lo, hi);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto addI16(VectorI16 a, VectorI16 b) -> VectorI16 {
#if defined(__AVX512F__)
  return _mm512_add_epi16(a, b);
#elif defined(__AVX2__)
  return _mm256_add_epi16(a, b);
#elif defined(__SSE4_1__)
  return _mm_add_epi16(a, b);
#else
  return static_cast<VectorI16>(a + b);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto subI16(VectorI16 a, VectorI16 b) -> VectorI16 {
#if defined(__AVX512F__)
  return _mm512_sub_epi16(a, b);
#elif defined(__AVX2__)
  return _mm256_sub_epi16(a, b);
#elif defined(__SSE4_1__)
  return _mm_sub_epi16(a, b);
#else
  return static_cast<VectorI16>(a - b);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto mulI16(VectorI16 a, VectorI16 b) -> VectorI16 {
#if defined(__AVX512F__)
  return _mm512_mullo_epi16(a, b);
#elif defined(__AVX2__)
  return _mm256_mullo_epi16(a, b);
#elif defined(__SSE4_1__)
  return _mm_mullo_epi16(a, b);
#else
  return static_cast<VectorI16>(a * b);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto mulAddAdjI16(VectorI16 a, VectorI16 b) -> VectorI32 {
#if defined(__AVX512F__)
  return _mm512_madd_epi16(a, b);
#elif defined(__AVX2__)
  return _mm256_madd_epi16(a, b);
#elif defined(__SSE4_1__)
  return _mm_madd_epi16(a, b);
#else
  return static_cast<VectorI32>(a) * static_cast<VectorI32>(b);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto zeroI32() -> VectorI32 {
#if defined(__AVX512F__)
  return _mm512_setzero_si512();
#elif defined(__AVX2__)
  return _mm256_setzero_si256();
#elif defined(__SSE4_1__)
  return _mm_setzero_si128();
#else
  return 0;
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto set1I32(i32 v) -> VectorI32 {
#if defined(__AVX512F__)
  return _mm512_set1_epi32(v);
#elif defined(__AVX2__)
  return _mm256_set1_epi32(v);
#elif defined(__SSE4_1__)
  return _mm_set1_epi32(v);
#else
  return v;
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto loadI32(const void *ptr) -> VectorI32 {
  assert(isAligned(ptr));
#if defined(__AVX512F__)
  return _mm512_load_si512(ptr);
#elif defined(__AVX2__)
  return _mm256_load_si256(static_cast<const VectorI16 *>(ptr));
#elif defined(__SSE4_1__)
  return _mm_load_si128(static_cast<const VectorI16 *>(ptr));
#else
  return *static_cast<const VectorI32 *>(ptr);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto storeI32(void *ptr, VectorI32 v) -> void {
  assert(isAligned(ptr));
#if defined(__AVX512F__)
  _mm512_store_si512(ptr, v);
#elif defined(__AVX2__)
  _mm256_store_si256(static_cast<VectorI32 *>(ptr), v);
#elif defined(__SSE4_1__)
  _mm_store_si128(static_cast<VectorI32 *>(ptr), v);
#else
  *static_cast<VectorI32 *>(ptr) = v;
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto minI32(VectorI32 a, VectorI32 b) -> VectorI32 {
#if defined(__AVX512F__)
  return _mm512_min_epi32(a, b);
#elif defined(__AVX2__)
  return _mm256_min_epi32(a, b);
#elif defined(__SSE4_1__)
  return _mm_min_epi32(a, b);
#else
  return std::min(a, b);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto maxI32(VectorI32 a, VectorI32 b) -> VectorI32 {
#if defined(__AVX512F__)
  return _mm512_max_epi32(a, b);
#elif defined(__AVX2__)
  return _mm256_max_epi32(a, b);
#elif defined(__SSE4_1__)
  return _mm_max_epi32(a, b);
#else
  return std::max(a, b);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto clampI32(VectorI32 v, VectorI32 lo, VectorI32 hi) -> VectorI32 {
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE4_1__)
  return minI32(maxI32(v, lo), hi);
#else
  return std::clamp(v, lo, hi);
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto addI32(VectorI32 a, VectorI32 b) -> VectorI32 {
#if defined(__AVX512F__)
  return _mm512_add_epi32(a, b);
#elif defined(__AVX2__)
  return _mm256_add_epi32(a, b);
#elif defined(__SSE4_1__)
  return _mm_add_epi32(a, b);
#else
  return a + b;
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto subI32(VectorI32 a, VectorI32 b) -> VectorI32 {
#if defined(__AVX512F__)
  return _mm512_sub_epi32(a, b);
#elif defined(__AVX2__)
  return _mm256_sub_epi32(a, b);
#elif defined(__SSE4_1__)
  return _mm_sub_epi32(a, b);
#else
  return a - b;
#endif
}

EVAL_ALWAYS_INLINE_NDEBUG inline auto mulI32(VectorI32 a, VectorI32 b) -> VectorI32 {
#if defined(__AVX512F__)
  return _mm512_mullo_epi32(a, b);
#elif defined(__AVX2__)
  return _mm256_mullo_epi32(a, b);
#elif defined(__SSE4_1__)
  return _mm_mullo_epi32(a, b);
#else
  return a * b;
#endif
}

namespace internal {
#if defined(__SSE4_1__)
EVAL_ALWAYS_INLINE_NDEBUG inline auto hsumI32Sse41(__m128i v) -> i32 {
  const auto high64 = _mm_unpackhi_epi64(v, v);
  const auto sum64 = _mm_add_epi32(v, high64);
  const auto high32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  const auto sum32 = _mm_add_epi32(sum64, high32);
  return _mm_cvtsi128_si32(sum32);
}
#endif

#if defined(__AVX2__)
EVAL_ALWAYS_INLINE_NDEBUG inline auto hsumI32Avx2(__m256i v) -> i32 {
  const auto high128 = _mm256_extracti128_si256(v, 1);
  const auto low128 = _mm256_castsi256_si128(v);
  const auto sum128 = _mm_add_epi32(high128, low128);
  return hsumI32Sse41(sum128);
}
#endif

#if defined(__AVX512F__)
EVAL_ALWAYS_INLINE_NDEBUG inline auto hsumI32Avx512(__m512i v) -> i32 {
  const auto high256 = _mm512_extracti64x4_epi64(v, 1);
  const auto low256 = _mm512_castsi512_si256(v);
  const auto sum256 = _mm256_add_epi32(high256, low256);
  return hsumI32Avx2(sum256);
}
#endif
}  // namespace internal

EVAL_ALWAYS_INLINE_NDEBUG inline auto hsumI32(VectorI32 v) -> i32 {
#if defined(__AVX512F__)
  return internal::hsumI32Avx512(v);
#elif defined(__AVX2__)
  return internal::hsumI32Avx2(v);
#elif defined(__SSE4_1__)
  return internal::hsumI32Sse41(v);
#else
  return v;
#endif
}
}  // namespace impl

template <typename T>
struct VectorImpl {};

template <>
struct VectorImpl<i16> {
  using Type = VectorI16;
};

template <>
struct VectorImpl<i32> {
  using Type = VectorI32;
};

template <typename T>
using Vector = typename VectorImpl<T>::Type;

#define EVAL_SIMD_OP_0(Name)                 \
  template <typename T>                      \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name() = delete; \
  template <>                                \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name<i16>() { return impl::Name##I16(); } \
  template <>                                \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name<i32>() { return impl::Name##I32(); }

#define EVAL_SIMD_OP_1_VALUE(Name, Arg0)     \
  template <typename T>                      \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name(T Arg0) = delete; \
  template <>                                \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name<i16>(i16 Arg0) { return impl::Name##I16(Arg0); } \
  template <>                                \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name<i32>(i32 Arg0) { return impl::Name##I32(Arg0); }

#define EVAL_SIMD_OP_2_VECTORS(Name, Arg0, Arg1) \
  template <typename T>                           \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name(Vector<T> Arg0, Vector<T> Arg1) = delete; \
  template <>                                     \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name<i16>(Vector<i16> Arg0, Vector<i16> Arg1) { \
    return impl::Name##I16(Arg0, Arg1); \
  }                                               \
  template <>                                     \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name<i32>(Vector<i32> Arg0, Vector<i32> Arg1) { \
    return impl::Name##I32(Arg0, Arg1); \
  }

#define EVAL_SIMD_OP_3_VECTORS(Name, Arg0, Arg1, Arg2) \
  template <typename T>                                 \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name(Vector<T> Arg0, Vector<T> Arg1, Vector<T> Arg2) = delete; \
  template <>                                           \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name<i16>(Vector<i16> Arg0, Vector<i16> Arg1, Vector<i16> Arg2) { \
    return impl::Name##I16(Arg0, Arg1, Arg2); \
  }                                                   \
  template <>                                          \
  EVAL_ALWAYS_INLINE_NDEBUG inline auto Name<i32>(Vector<i32> Arg0, Vector<i32> Arg1, Vector<i32> Arg2) { \
    return impl::Name##I32(Arg0, Arg1, Arg2); \
  }

EVAL_SIMD_OP_0(zero)
EVAL_SIMD_OP_1_VALUE(set1, v)
EVAL_SIMD_OP_2_VECTORS(add, a, b)
EVAL_SIMD_OP_2_VECTORS(sub, a, b)
EVAL_SIMD_OP_2_VECTORS(mul, a, b)
EVAL_SIMD_OP_2_VECTORS(min, a, b)
EVAL_SIMD_OP_2_VECTORS(max, a, b)
EVAL_SIMD_OP_3_VECTORS(clamp, v, lo, hi)

template <typename T>
EVAL_ALWAYS_INLINE_NDEBUG inline auto load(const void *ptr) = delete;
template <>
EVAL_ALWAYS_INLINE_NDEBUG inline auto load<i16>(const void *ptr) { return impl::loadI16(ptr); }
template <>
EVAL_ALWAYS_INLINE_NDEBUG inline auto load<i32>(const void *ptr) { return impl::loadI32(ptr); }

template <typename T>
EVAL_ALWAYS_INLINE_NDEBUG inline auto store(void *ptr, Vector<T> v) = delete;
template <>
EVAL_ALWAYS_INLINE_NDEBUG inline auto store<i16>(void *ptr, Vector<i16> v) { impl::storeI16(ptr, v); }
template <>
EVAL_ALWAYS_INLINE_NDEBUG inline auto store<i32>(void *ptr, Vector<i32> v) { impl::storeI32(ptr, v); }

template <typename T>
EVAL_ALWAYS_INLINE_NDEBUG inline auto mulAddAdj(Vector<T> a, Vector<T> b) = delete;
template <>
EVAL_ALWAYS_INLINE_NDEBUG inline auto mulAddAdj<i16>(Vector<i16> a, Vector<i16> b) { return impl::mulAddAdjI16(a, b); }

template <typename T>
EVAL_ALWAYS_INLINE_NDEBUG inline auto hsum(Vector<T> v) = delete;
template <>
EVAL_ALWAYS_INLINE_NDEBUG inline auto hsum<i32>(Vector<i32> v) { return impl::hsumI32(v); }

#undef EVAL_SIMD_OP_0
#undef EVAL_SIMD_OP_1_VALUE
#undef EVAL_SIMD_OP_2_VECTORS
#undef EVAL_SIMD_OP_3_VECTORS
}  // namespace eval::util::simd
