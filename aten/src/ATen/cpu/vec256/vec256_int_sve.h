#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if defined(__GNUC__) && defined(__ARM_FEATURE_SVE)

#define VEC256_INT_SVE_TEMPLATE(vl, bit)				                                \
template <> class Vec256<int##bit##_t> {                                                                \
private:                                                                                                \
  /* TODO: Convert to svintXX_t. */                                                                     \
  __at_align__ int##bit##_t values[vl];                                                                 \
public:                                                                                                 \
  using value_type = int##bit##_t;                                                                      \
  static constexpr int size() {                                                                         \
    return vl;                                                                                          \
  }                                                                                                     \
  Vec256(const Vec256& rhs) {                                                                           \
    *reinterpret_cast<svint##bit##_t*>(values) = *reinterpret_cast<const svint##bit##_t*>(rhs.values);  \
  }                                                                                                     \
  Vec256& operator=(const Vec256& rhs) {                                                                \
    *reinterpret_cast<svint##bit##_t*>(values) = *reinterpret_cast<const svint##bit##_t*>(rhs.values);  \
    return *this;                                                                                       \
  }                                                                                                     \
  Vec256() {}                                                                                           \
  Vec256(svint##bit##_t v) {                                                                            \
    *reinterpret_cast<svint##bit##_t*>(values) = v;                                                     \
  }                                                                                                     \
  Vec256(int##bit##_t val) {                                                                            \
    *reinterpret_cast<svint##bit##_t*>(values) = svdup_n_s##bit(val);                                   \
  }                                                                                                     \
  template<typename... Args,                                                                            \
           typename = std::enable_if_t<(sizeof...(Args) == size())>>                                    \
  Vec256(Args... vals) {                                                                                \
    __at_align__ int##bit##_t buffer[size()] = { vals... };                                             \
    *reinterpret_cast<svint##bit##_t*>(values) = svld1_s##bit(ptrue, buffer);                           \
  }                                                                                                     \
  operator svint##bit##_t() const {                                                                     \
    return *reinterpret_cast<const svint##bit##_t*>(values);                                            \
  }                                                                                                     \
  static Vec256<int##bit##_t> blendv(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b,      \
				const Vec256<int##bit##_t>& mask_) {                                    \
    svbool_t mask = svcmpeq_s##bit(ptrue, mask_, ALL_S##bit##_TRUE_MASK);                               \
    return svsel_s##bit(mask, b, a);                                                                    \
  }                                                                                                     \
  template <typename step_t>                                                                            \
  static Vec256<int##bit##_t> arange(int##bit##_t base = 0, step_t step = static_cast<step_t>(1)) {     \
    return svindex_s##bit(base, step);                                                                  \
  }                                                                                                     \
  static Vec256<int##bit##_t> set(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b,         \
                           int##bit##_t count = size()) {                                               \
    if (count == 0) {                                                                                   \
      return a;                                                                                         \
    } else if (count < size()) {                                                                        \
      return svsel_s##bit(svwhilelt_b##bit(0ull, count), b, a);                                         \
    }                                                                                                   \
    return b;                                                                                           \
  }                                                                                                     \
  static Vec256<int##bit##_t> loadu(const void* ptr, int64_t count = size()) {                          \
    if (count == size())                                                                                \
      return svld1_s##bit(ptrue, reinterpret_cast<const int##bit##_t*>(ptr));                           \
    svbool_t pg = svwhilelt_b##bit(0ull, count);                                                        \
    return svld1_s##bit(pg, reinterpret_cast<const int##bit##_t*>(ptr));                                \
  }                                                                                                     \
  void store(void* ptr, int64_t count = size()) const {                                                 \
    if (count == size()) {                                                                              \
      svst1_s##bit(ptrue, reinterpret_cast<int##bit##_t*>(ptr), *this);                                 \
    } else {                                                                                            \
      svbool_t pg = svwhilelt_b##bit(0ull, count);                                                      \
      svst1_s##bit(pg, reinterpret_cast<int##bit##_t*>(ptr), *this);                                    \
    }                                                                                                   \
  }                                                                                                     \
  const int##bit##_t& operator[](int idx) const  = delete;                                              \
  int##bit##_t& operator[](int idx) = delete;                                                           \
  Vec256<int##bit##_t> abs() const {                                                                    \
    return svabs_s##bit##_x(ptrue, *this);                                                              \
  }                                                                                                     \
  Vec256<int##bit##_t> angle() const {                                                                  \
    return svdup_n_s##bit(0);					                     	                \
  }                                                                                                     \
  Vec256<int##bit##_t> real() const {                                                                   \
    return *this;                                                                                       \
  }                                                                                                     \
  Vec256<int##bit##_t> imag() const {                                                                   \
    return svdup_n_s##bit(0);                                                                           \
  }                                                                                                     \
  Vec256<int##bit##_t> conj() const {                                                                   \
    return *this;                                                                                       \
  }                                                                                                     \
  Vec256<int##bit##_t> frac() const;                                                                    \
  Vec256<int##bit##_t> neg() const {                                                                    \
    return svneg_s##bit##_x(ptrue, *this);                                                              \
  }                                                                                                     \
  Vec256<int##bit##_t> operator==(const Vec256<int##bit##_t>& other) const {                            \
    svbool_t mask = svcmpeq_s##bit(ptrue, *this, other);                                                \
    return svsel_s##bit(mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);                         \
  }                                                                                                     \
  Vec256<int##bit##_t> operator!=(const Vec256<int##bit##_t>& other) const {                            \
    svbool_t mask = svcmpne_s##bit(ptrue, *this, other);                                                \
    return svsel_s##bit(mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);                         \
  }                                                                                                     \
  Vec256<int##bit##_t> operator<(const Vec256<int##bit##_t>& other) const {                             \
    svbool_t mask = svcmplt_s##bit(ptrue, *this, other);                                                \
    return svsel_s##bit(mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);                         \
  }                                                                                                     \
  Vec256<int##bit##_t> operator<=(const Vec256<int##bit##_t>& other) const {                            \
    svbool_t mask = svcmple_s##bit(ptrue, *this, other);                                                \
    return svsel_s##bit(mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);                         \
  }                                                                                                     \
  Vec256<int##bit##_t> operator>(const Vec256<int##bit##_t>& other) const {                             \
    svbool_t mask = svcmpgt_s##bit(ptrue, *this, other);                                                \
    return svsel_s##bit(mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);                         \
  }                                                                                                     \
  Vec256<int##bit##_t> operator>=(const Vec256<int##bit##_t>& other) const {                            \
    svbool_t mask = svcmpge_s##bit(ptrue, *this, other);                                                \
    return svsel_s##bit(mask, ALL_S##bit##_TRUE_MASK, ALL_S##bit##_FALSE_MASK);                         \
  }                                                                                                     \
  Vec256<int##bit##_t> eq(const Vec256<int##bit##_t>& other) const;                                     \
  Vec256<int##bit##_t> ne(const Vec256<int##bit##_t>& other) const;                                     \
  Vec256<int##bit##_t> gt(const Vec256<int##bit##_t>& other) const;                                     \
  Vec256<int##bit##_t> ge(const Vec256<int##bit##_t>& other) const;                                     \
  Vec256<int##bit##_t> lt(const Vec256<int##bit##_t>& other) const;                                     \
  Vec256<int##bit##_t> le(const Vec256<int##bit##_t>& other) const;                                     \
};                                                                                                      \
template <>                                                                                             \
Vec256<int##bit##_t> inline operator+(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b) {   \
  return svadd_s##bit##_x(ptrue, a, b);                                                                 \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline operator-(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b) {   \
  return svsub_s##bit##_x(ptrue, a, b);                                                                 \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline operator*(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b) {   \
  return svmul_s##bit##_x(ptrue, a, b);                                                                 \
}                                                                                                       \
template <>								                                \
Vec256<int##bit##_t> inline maximum(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b) {     \
  return svmax_s##bit##_x(ptrue, a, b);                                                                 \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline minimum(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b) {     \
  return svmin_s##bit##_x(ptrue, a, b);                                                                 \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline clamp(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& min,       \
			  const Vec256<int##bit##_t>& max) {                                            \
  return svmin_s##bit##_x(ptrue, max, svmax_s##bit##_x(ptrue, min, a));                                 \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline clamp_max(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& max) { \
  return svmin_s##bit##_x(ptrue, max, a);                                                               \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline clamp_min(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& min) { \
  return svmax_s##bit##_x(ptrue, min, a);                                                               \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline operator&(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b) {   \
  return svand_s##bit##_x(ptrue, a, b);                                                                 \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline operator|(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b) {   \
  return svorr_s##bit##_x(ptrue, a, b);                                                                 \
}                                                                                                       \
template <>                                                                                             \
Vec256<int##bit##_t> inline operator^(const Vec256<int##bit##_t>& a, const Vec256<int##bit##_t>& b) {   \
  return sveor_s##bit##_x(ptrue, a, b);                                                                 \
}                                                                                                       \
template <>                                                                                             \
inline Vec256<int##bit##_t> operator~(const Vec256<int##bit##_t>& a) {                                  \
  return sveor_s##bit##_x(ptrue, a, svdup_n_s##bit(-1));                                                \
}                                                                                                       \
Vec256<int##bit##_t> Vec256<int##bit##_t>::eq(const Vec256<int##bit##_t>& other) const {                \
  return (*this == other) & Vec256<int##bit##_t>(1);                                                    \
}                                                                                                       \
Vec256<int##bit##_t> Vec256<int##bit##_t>::ne(const Vec256<int##bit##_t>& other) const {                \
  return (*this != other) & Vec256<int##bit##_t>(1);                                                    \
}                                                                                                       \
Vec256<int##bit##_t> Vec256<int##bit##_t>::gt(const Vec256<int##bit##_t>& other) const {                \
  return (*this > other) & Vec256<int##bit##_t>(1);                                                     \
}                                                                                                       \
Vec256<int##bit##_t> Vec256<int##bit##_t>::ge(const Vec256<int##bit##_t>& other) const {                \
  return (*this >= other) & Vec256<int##bit##_t>(1);                                                    \
}                                                                                                       \
Vec256<int##bit##_t> Vec256<int##bit##_t>::lt(const Vec256<int##bit##_t>& other) const {                \
  return (*this < other) & Vec256<int##bit##_t>(1);                                                     \
}                                                                                                       \
Vec256<int##bit##_t> Vec256<int##bit##_t>::le(const Vec256<int##bit##_t>& other) const {                \
  return (*this <= other) & Vec256<int##bit##_t>(1);                                                    \
}

VEC256_INT_SVE_TEMPLATE(8, 64)
VEC256_INT_SVE_TEMPLATE(16, 32)
VEC256_INT_SVE_TEMPLATE(32, 16)
VEC256_INT_SVE_TEMPLATE(64, 8)

template <typename T>
Vec256<T> inline intdiv_nosve(const Vec256<T>& a, const Vec256<T>& b) {
  T values_a[Vec256<T>::size()];
  T values_b[Vec256<T>::size()];
  a.store(values_a);
  b.store(values_b);
  for (int i = 0; i != Vec256<T>::size(); i++) {
    values_a[i] /= values_b[i];
  }
  return Vec256<T>::loadu(values_a);
}

template <>
Vec256<int64_t> inline operator/(const Vec256<int64_t>& a, const Vec256<int64_t>& b) {
  return svdiv_s64_x(ptrue, a, b);
}

template <>
Vec256<int32_t> inline operator/(const Vec256<int32_t>& a, const Vec256<int32_t>& b) {
  return svdiv_s32_x(ptrue, a, b);
}

template <>
Vec256<int16_t> inline operator/(const Vec256<int16_t>& a, const Vec256<int16_t>& b) {
  return intdiv_nosve(a, b);
}

template <>
Vec256<int8_t> inline operator/(const Vec256<int8_t>& a, const Vec256<int8_t>& b) {
  return intdiv_nosve(a, b);
}

template <>
inline void convert(const int32_t *src, int64_t *dst, int64_t n) {
  const int64_t fraction = n % Vec256<int64_t>::size();
  svbool_t pg_32 = svwhilelt_b32(0ull, Vec256<int64_t>::size());
  svbool_t pg_64 = svwhilelt_b64(0ull, Vec256<int64_t>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vec256<int64_t>::size())
    svst1_s64(pg_64, dst + i, svunpklo_s64(svldnt1_s32(pg_32, src + i)));
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vec256<int64_t>::size()) {
    pg_32 = svwhilelt_b32(i, n);
    pg_64 = svwhilelt_b64(i, n);
    svst1_s64(pg_64, dst + i, svunpklo_s64(svldnt1_s32(pg_32, src + i)));
  }
}

template <>
inline void convert(const int64_t *src, float *dst, int64_t n) {
  const int64_t fraction = n % Vec256<int64_t>::size();
  svbool_t pg_32 = svwhilelt_b32(0ull, Vec256<int64_t>::size());
  svbool_t pg_64 = svwhilelt_b64(0ull, Vec256<int64_t>::size());
  svfloat32_t zero_f32 = svdup_n_f32(0.f);
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vec256<int64_t>::size()) {
    svint64_t src_vec_s64 = svldnt1_s64(pg_64, src + i);
    svfloat32_t src_vec_f32 = svuzp1_f32(svcvt_f32_s64_x(pg_64, src_vec_s64), zero_f32);
    svst1_f32(pg_32, dst + i, src_vec_f32);
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vec256<int64_t>::size()) {
    pg_32 = svwhilelt_b32(i, n);
    pg_64 = svwhilelt_b64(i, n);
    svint64_t src_vec_s64 = svldnt1_s64(pg_64, src + i);
    svfloat32_t src_vec_f32 = svuzp1_f32(svcvt_f32_s64_x(pg_64, src_vec_s64), zero_f32);
    svst1_f32(pg_32, dst + i, src_vec_f32);
  }
}

template <>
inline void convert(const int32_t *src, float *dst, int64_t n) {
  const int64_t fraction = n % Vec256<int32_t>::size();
  svbool_t pg = svwhilelt_b32(0ull, Vec256<int32_t>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vec256<int32_t>::size()) {
    svint32_t src_vec = svldnt1_s32(pg, src + i);
    svst1_f32(pg, dst + i, svcvt_f32_s32_x(pg, src_vec));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vec256<int32_t>::size()) {
    pg = svwhilelt_b32(i, n);
    svint32_t src_vec = svldnt1_s32(pg, src + i);
    svst1_f32(pg, dst + i, svcvt_f32_s32_x(pg, src_vec));
  }
}

template <>
inline void convert(const bool *src, int64_t *dst, int64_t n) {
  const int64_t fraction = n % Vec256<int64_t>::size();
  svbool_t pg_8 = svwhilelt_b8(0ull, Vec256<int64_t>::size());
  svbool_t pg_64 = svwhilelt_b64(0ull, Vec256<int64_t>::size());
  svuint64_t zero_u64 = svdup_n_u64(0);
  svint64_t zero_s64 = svdup_n_s64(0);
  svint64_t one_s64 = svdup_n_s64(1);
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vec256<int64_t>::size()) {
    svuint8_t src_vec_u8 = svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint64_t src_vec_u64 = svunpklo_u64(svunpklo_u32(svunpklo_u16(src_vec_u8)));
    svbool_t mask = svcmpne_u64(pg_64, src_vec_u64, zero_u64);
    svst1_s64(pg_64, dst + i, svsel_s64(mask, one_s64, zero_s64));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vec256<int64_t>::size()) {
    pg_8 = svwhilelt_b8(i, n);
    pg_64 = svwhilelt_b64(i, n);
    svuint8_t src_vec_u8 = svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint64_t src_vec_u64 = svunpklo_u64(svunpklo_u32(svunpklo_u16(src_vec_u8)));
    svbool_t mask = svcmpne_u64(pg_64, src_vec_u64, zero_u64);
    svst1_s64(pg_64, dst + i, svsel_s64(mask, one_s64, zero_s64));
  }
}

template <>
inline void convert(const bool *src, int32_t *dst, int64_t n) {
  const int64_t fraction = n % Vec256<int32_t>::size();
  svbool_t pg_8 = svwhilelt_b8(0ull, Vec256<int32_t>::size());
  svbool_t pg_32 = svwhilelt_b32(0ull, Vec256<int32_t>::size());
  svuint32_t zero_u32 = svdup_n_u32(0);
  svint32_t zero_s32 = svdup_n_s32(0);
  svint32_t one_s32 = svdup_n_s32(1);
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vec256<int32_t>::size()) {
    svuint8_t src_vec_u8 = svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, zero_u32);
    svst1_s32(pg_32, dst + i, svsel_s32(mask, one_s32, zero_s32));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vec256<int32_t>::size()) {
    pg_8 = svwhilelt_b8(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svuint8_t src_vec_u8 = svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, zero_u32);
    svst1_s32(pg_32, dst + i, svsel_s32(mask, one_s32, zero_s32));
  }
}

template <>
inline void convert(const uint8_t *src, bool *dst, int64_t n) {
  const int64_t fraction = n % Vec256<uint8_t>::size();
  svbool_t pg = svwhilelt_b8(0ull, Vec256<uint8_t>::size());
  svuint8_t zero_u8 = svdup_n_u8(0);
  svuint8_t false_vec = svdup_n_u8(0x00);
  svuint8_t true_vec = svdup_n_u8(0x01);
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vec256<uint8_t>::size()) {
    svbool_t mask = svcmpne_u8(pg, svldnt1_u8(pg, src + i), zero_u8);
    svst1_u8(pg, reinterpret_cast<uint8_t*>(dst) + i,
	     svsel_u8(mask, true_vec, false_vec));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vec256<uint8_t>::size()) {
    pg = svwhilelt_b8(i, n);
    svbool_t mask = svcmpne_u8(pg, svldnt1_u8(pg, src + i), zero_u8);
    svst1_u8(pg, reinterpret_cast<uint8_t*>(dst) + i,
             svsel_u8(mask, true_vec, false_vec));
  }
}

#endif

}}}
