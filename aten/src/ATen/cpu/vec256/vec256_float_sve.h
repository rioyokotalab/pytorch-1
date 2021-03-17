#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#if defined(__GNUC__) && defined(__ARM_FEATURE_SVE)
#include <sleef.h>
#endif

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if defined(__GNUC__) && defined(__ARM_FEATURE_SVE)

template <> class Vec256<float> {
private:
  /* TODO: Convert to svfloat32_t. */
  __at_align__ float values[16];
public:
  using value_type = float;
  static constexpr int size() {
    return 16;
  }
  Vec256(const Vec256& rhs) {
    *reinterpret_cast<svfloat32_t*>(values) = *reinterpret_cast<const svfloat32_t*>(rhs.values);
  }
  Vec256& operator=(const Vec256& rhs) {
    *reinterpret_cast<svfloat32_t*>(values) = *reinterpret_cast<const svfloat32_t*>(rhs.values);
    return *this;
  }
  Vec256() {}
  Vec256(svfloat32_t v) {
    *reinterpret_cast<svfloat32_t*>(values) = v;
  }
  Vec256(float val) {
    *reinterpret_cast<svfloat32_t*>(values) = svdup_n_f32(val);
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vec256(Args... vals) {
    __at_align__ float buffer[size()] = { vals... };
    *reinterpret_cast<svfloat32_t*>(values) = svld1_f32(ptrue, buffer);
  }
  operator svfloat32_t() const {
    return *reinterpret_cast<const svfloat32_t*>(values);
  }
  static Vec256<float> blendv(const Vec256<float>& a, const Vec256<float>& b,
                              const Vec256<float>& mask_) {
    svbool_t mask = svcmpeq_s32(ptrue, svreinterpret_s32_f32(mask_),
				ALL_S32_TRUE_MASK);
    return svsel_f32(mask, b, a);
  }
  template<typename step_t>
  static Vec256<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    __at_align__ float buffer[size()];
    for (int64_t i = 0; i < size(); i++) {
      buffer[i] = base + i * step;
    }
    return svld1_f32(ptrue, buffer);
  }
  static Vec256<float> set(const Vec256<float>& a, const Vec256<float>& b,
                           int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_f32(svwhilelt_b32(0ull, count), b, a);
    }
    return b;
  }
  static Vec256<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return svld1_f32(ptrue, reinterpret_cast<const float*>(ptr));
    svbool_t pg = svwhilelt_b32(0ull, count);
    return svld1_f32(pg, reinterpret_cast<const float*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      svst1_f32(ptrue, reinterpret_cast<float*>(ptr), *this);
    } else {
      svbool_t pg = svwhilelt_b32(0ull, count);
      svst1_f32(pg, reinterpret_cast<float*>(ptr), *this);
    }
  }
  const float& operator[](int idx) const  = delete;
  float& operator[](int idx) = delete;
  int64_t zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int64_t mask = 0;
    int32_t mask_array[svcntw()];

    svbool_t svbool_mask = svcmpeq_f32(ptrue, *this, svdup_n_f32(0.f));
    svst1_s32(ptrue, mask_array, svsel_s32(svbool_mask,
					   ALL_S32_TRUE_MASK,
					   ALL_S32_FALSE_MASK));
    for (int64_t i = 0; i < svcntw(); ++i) {
      if (mask_array[i]) mask |= (1ull << i);
    }
    return mask;
  }
  Vec256<float> map(float (*f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); ++i) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> abs() const {
    return svabs_f32_x(ptrue, *this);
  }
  Vec256<float> angle() const {
    return Vec256<float>(0.f);
  }
  Vec256<float> real() const {
    return *this;
  }
  Vec256<float> imag() const {
    return Vec256<float>(0.f);
  }
  Vec256<float> conj() const {
    return *this;
  }
  Vec256<float> acos() const {
    return Vec256<float>(Sleef_acosfx_u10sve(*this));
  }
  Vec256<float> asin() const {
    return Vec256<float>(Sleef_asinfx_u10sve(*this));
  }
  Vec256<float> atan() const {
    return Vec256<float>(Sleef_atanfx_u10sve(*this));
  }
  Vec256<float> atan2(const Vec256<float> &b) const {
    return Vec256<float>(Sleef_atan2fx_u10sve(*this, b));
  }
  Vec256<float> erf() const {
    return Vec256<float>(Sleef_erffx_u10sve(*this));
  }
  Vec256<float> erfc() const {
    return Vec256<float>(Sleef_erfcfx_u15sve(*this));
  }
  Vec256<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vec256<float> exp() const {
    return Vec256<float>(Sleef_expfx_u10sve(*this));
  }
  Vec256<float> expm1() const {
    return Vec256<float>(Sleef_expm1fx_u10sve(*this));
  }
  Vec256<float> fmod(const Vec256<float>& q) const {
    return Vec256<float>(Sleef_fmodfx_sve(*this, q));
  }
  Vec256<float> hypot(const Vec256<float> &b) const {
    return Vec256<float>(Sleef_hypotfx_u05sve(*this, b));
  }
  Vec256<float> i0() const {
    return map(calc_i0);
  }
  Vec256<float> nextafter(const Vec256<float> &b) const {
    return Vec256<float>(Sleef_nextafterfx_sve(*this, b));
  }
  Vec256<float> log() const {
    return Vec256<float>(Sleef_logfx_u10sve(*this));
  }
  Vec256<float> log2() const {
    return Vec256<float>(Sleef_log2fx_u10sve(*this));
  }
  Vec256<float> log10() const {
    return Vec256<float>(Sleef_log10fx_u10sve(*this));
  }
  Vec256<float> log1p() const {
    return Vec256<float>(Sleef_log1pfx_u10sve(*this));
  }
  Vec256<float> frac() const;
  Vec256<float> sin() const {
    return Vec256<float>(Sleef_sinfx_u10sve(*this));
  }
  Vec256<float> sinh() const {
    return Vec256<float>(Sleef_sinhfx_u10sve(*this));
  }
  Vec256<float> cos() const {
    return Vec256<float>(Sleef_cosfx_u10sve(*this));
  }
  Vec256<float> cosh() const {
    return Vec256<float>(Sleef_coshfx_u10sve(*this));
  }
  Vec256<float> ceil() const {
    return svrintp_f32_x(ptrue, *this);
  }
  Vec256<float> floor() const {
    return svrintm_f32_x(ptrue, *this);
  }
  Vec256<float> neg() const {
    return svneg_f32_x(ptrue, *this);
  }
  Vec256<float> round() const {
    return svrinti_f32_x(ptrue, *this);
  }
  Vec256<float> tan() const {
    return Vec256<float>(Sleef_tanfx_u10sve(*this));
  }
  Vec256<float> tanh() const {
    return Vec256<float>(Sleef_tanhfx_u10sve(*this));
  }
  Vec256<float> trunc() const {
    return svrintz_f32_x(ptrue, *this);
  }
  Vec256<float> lgamma() const {
    return Vec256<float>(Sleef_lgammafx_u10sve(*this));
  }
  Vec256<float> sqrt() const {
    return svsqrt_f32_x(ptrue, *this);
  }
  Vec256<float> reciprocal() const {
    return svdivr_f32_x(ptrue, *this, svdup_n_f32((float)1.0));
  }
  Vec256<float> rsqrt() const {
    return svdivr_f32_x(ptrue, svsqrt_f32_x(ptrue, *this), svdup_n_f32((float)1.0));
  }
  Vec256<float> pow(const Vec256<float> &b) const {
    return Vec256<float>(Sleef_powfx_u10sve(*this, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<float> operator==(const Vec256<float>& other) const {
    svbool_t mask = svcmpeq_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vec256<float> operator!=(const Vec256<float>& other) const {
    svbool_t mask = svcmpne_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vec256<float> operator<(const Vec256<float>& other) const {
    svbool_t mask = svcmplt_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vec256<float> operator<=(const Vec256<float>& other) const {
    svbool_t mask = svcmple_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vec256<float> operator>(const Vec256<float>& other) const {
    svbool_t mask = svcmpgt_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vec256<float> operator>=(const Vec256<float>& other) const {
    svbool_t mask = svcmpge_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vec256<float> eq(const Vec256<float>& other) const;
  Vec256<float> ne(const Vec256<float>& other) const;
  Vec256<float> gt(const Vec256<float>& other) const;
  Vec256<float> ge(const Vec256<float>& other) const;
  Vec256<float> lt(const Vec256<float>& other) const;
  Vec256<float> le(const Vec256<float>& other) const;
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  return svadd_f32_x(ptrue, a, b);
}

template <>
Vec256<float> inline operator-(const Vec256<float>& a, const Vec256<float>& b) {
  return svsub_f32_x(ptrue, a, b);
}

template <>
Vec256<float> inline operator*(const Vec256<float>& a, const Vec256<float>& b) {
  return svmul_f32_x(ptrue, a, b);
}

template <>
Vec256<float> inline operator/(const Vec256<float>& a, const Vec256<float>& b) {
  return svdiv_f32_x(ptrue, a, b);
}

// frac. Implement this here so we can use subtraction
Vec256<float> Vec256<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<float> inline maximum(const Vec256<float>& a, const Vec256<float>& b) {
  return svmax_f32_x(ptrue, a, b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<float> inline minimum(const Vec256<float>& a, const Vec256<float>& b) {
  return svmin_f32_x(ptrue, a, b);
}

template <>
Vec256<float> inline clamp(const Vec256<float>& a, const Vec256<float>& min, const Vec256<float>& max) {
  return svminnm_f32_x(ptrue, max, svmaxnm_f32_x(ptrue, min, a));
}

template <>
Vec256<float> inline clamp_max(const Vec256<float>& a, const Vec256<float>& max) {
  return svminnm_f32_x(ptrue, max, a);
}

template <>
Vec256<float> inline clamp_min(const Vec256<float>& a, const Vec256<float>& min) {
  return svmaxnm_f32_x(ptrue, min, a);
}

template <>
Vec256<float> inline operator&(const Vec256<float>& a, const Vec256<float>& b) {
  return svreinterpret_f32_s32(svand_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

template <>
Vec256<float> inline operator|(const Vec256<float>& a, const Vec256<float>& b) {
  return svreinterpret_f32_s32(svorr_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

template <>
Vec256<float> inline operator^(const Vec256<float>& a, const Vec256<float>& b) {
  return svreinterpret_f32_s32(sveor_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

Vec256<float> Vec256<float>::eq(const Vec256<float>& other) const {
  return (*this == other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::ne(const Vec256<float>& other) const {
  return (*this != other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::gt(const Vec256<float>& other) const {
  return (*this > other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::ge(const Vec256<float>& other) const {
  return (*this >= other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::lt(const Vec256<float>& other) const {
  return (*this < other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::le(const Vec256<float>& other) const {
  return (*this <= other) & Vec256<float>(1.0f);
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
#pragma unroll
  for (int64_t i = 0; i < n; i += Vec256<float>::size()) {
    svbool_t pg = svwhilelt_b32(i, n);
    svst1_f32(pg, dst + i, svldnt1_f32(pg, src + i));
  }
}

template <>
inline void convert(const float *src, at::Half *dst, int64_t n) {
#pragma unroll
  for (int64_t i = 0; i < n; i += Vec256<float>::size()) {
    svbool_t pg_16 = svwhilelt_b16(i, n);
    svbool_t pg_32 = svwhilelt_b32(i, n);
    svfloat16_t src_vec = svuzp1_f16(svcvt_f16_f32_x(ptrue, svldnt1_f32(pg_32, src + i)),
				     svdup_n_f16(0.0));
    svst1_f16(pg_16, reinterpret_cast<float16_t*>(dst) + i, src_vec);
  }
}

template <>
inline void convert(const at::Half *src, float *dst, int64_t n) {
#pragma unroll
  for (int64_t i = 0; i < n; i += Vec256<float>::size()) {
    svbool_t pg_16 = svwhilelt_b16(i, n);
    svbool_t pg_32 = svwhilelt_b32(i, n);
    svfloat16_t src_vec = svzip1_f16(svldnt1_f16(pg_16, reinterpret_cast<const float16_t*>(src) + i),
				     svdup_n_f16(0.0));
    svst1_f32(pg_32, dst + i, svcvt_f32_f16_x(ptrue, src_vec));
  }
}

template <>
Vec256<float> inline fmadd(const Vec256<float>& a, const Vec256<float>& b, const Vec256<float>& c) {
  return svmad_f32_x(ptrue, a, b, c);
}

#endif

}}}
