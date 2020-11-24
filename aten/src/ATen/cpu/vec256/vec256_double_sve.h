/*
 Copyright (c) 2020, FUJITSU LIMITED
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
   * Neither the name of the <organization> nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ==============================================================================*/

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

template <> class Vec256<double> {
private:
  /* TODO: Convert to svfloat64_t. */
  double values[8];
public:
  using value_type = double;
  static constexpr int size() {
    return 8;
  }
  Vec256(const Vec256& rhs) {
    *reinterpret_cast<svfloat64_t*>(values) = *reinterpret_cast<const svfloat64_t*>(rhs.values);
  }
  Vec256& operator=(const Vec256& rhs) {
    *reinterpret_cast<svfloat64_t*>(values) = *reinterpret_cast<const svfloat64_t*>(rhs.values);
    return *this;
  }
  Vec256() {}
  Vec256(svfloat64_t v) {
    *reinterpret_cast<svfloat64_t*>(values) = v;
  }
  Vec256(double val) {
    *reinterpret_cast<svfloat64_t*>(values) = svdup_n_f64(val);
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vec256(Args... vals) {
    double buffer[size()] = { vals... };
    *reinterpret_cast<svfloat64_t*>(values) = svld1_f64(ptrue, buffer);
  }
  operator svfloat64_t() const {
    return *reinterpret_cast<const svfloat64_t*>(values);
  }
  static Vec256<double> blendv(const Vec256<double>& a, const Vec256<double>& b,
                              const Vec256<double>& mask_) {
    svbool_t mask = svcmpeq_s64(ptrue, svreinterpret_s64_f64(mask_),
				ALL_S64_TRUE_MASK);
    return svsel_f64(mask, b, a);
  }
  template<typename step_t>
  static Vec256<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    double buffer[size()];
    for (int64_t i = 0; i < size(); i++) {
      buffer[i] = base + i * step;
    }
    return svld1_f64(ptrue, buffer);
  }
  static Vec256<double> set(const Vec256<double>& a, const Vec256<double>& b,
                           int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_f64(svwhilelt_b64(0ull, count), b, a);
    }
    return b;
  }
  static Vec256<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return svld1_f64(ptrue, reinterpret_cast<const double*>(ptr));
    svbool_t pg = svwhilelt_b64(0ull, count);
    return svld1_f64(pg, reinterpret_cast<const double*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      svst1_f64(ptrue, reinterpret_cast<double*>(ptr), *this);
    } else {
      svbool_t pg = svwhilelt_b64(0ull, count);
      svst1_f64(pg, reinterpret_cast<double*>(ptr), *this);
    }
  }
  const double& operator[](int idx) const  = delete;
  double& operator[](int idx) = delete;
  int64_t zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int64_t mask = 0;
    int64_t mask_array[svcntd()];

    svbool_t svbool_mask = svcmpeq_f64(ptrue, *this, svdup_n_f64(0.0));
    svst1_s64(ptrue, mask_array, svsel_s64(svbool_mask,
					   ALL_S64_TRUE_MASK,
					   ALL_S64_FALSE_MASK));
    for (int64_t i = 0; i < svcntd(); ++i) {
      if (mask_array[i]) mask |= (1ull << i);
    }
    return mask;
  }
  Vec256<double> map(double (*f)(double)) const {
    __at_align32__ double tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); ++i) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<double> abs() const {
    return svabs_f64_x(ptrue, *this);
  }
  Vec256<double> angle() const {
    return Vec256<double>(0.0);
  }
  Vec256<double> real() const {
    return *this;
  }
  Vec256<double> imag() const {
    return Vec256<double>(0.0);
  }
  Vec256<double> conj() const {
    return *this;
  }
  Vec256<double> acos() const {
    return Vec256<double>(Sleef_acosdx_u10sve(*this));
  }
  Vec256<double> asin() const {
    return Vec256<double>(Sleef_asindx_u10sve(*this));
  }
  Vec256<double> atan() const {
    return Vec256<double>(Sleef_atandx_u10sve(*this));
  }
  Vec256<double> atan2(const Vec256<double> &b) const {
    return Vec256<double>(Sleef_atan2dx_u10sve(*this, b));
  }
  Vec256<double> erf() const {
    return Vec256<double>(Sleef_erfdx_u10sve(*this));
  }
  Vec256<double> erfc() const {
    return Vec256<double>(Sleef_erfcdx_u15sve(*this));
  }
  Vec256<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vec256<double> exp() const {
    return Vec256<double>(Sleef_expdx_u10sve(*this));
  }
  Vec256<double> expm1() const {
    return Vec256<double>(Sleef_expm1dx_u10sve(*this));
  }
  Vec256<double> fmod(const Vec256<double>& q) const {
    return Vec256<double>(Sleef_fmoddx_sve(*this, q));
  }
  Vec256<double> hypot(const Vec256<double> &b) const {
    return Vec256<double>(Sleef_hypotdx_u05sve(*this, b));
  }
  Vec256<double> i0() const {
    return map(calc_i0);
  }
  Vec256<double> nextafter(const Vec256<double> &b) const {
    return Vec256<double>(Sleef_nextafterdx_sve(*this, b));
  }
  Vec256<double> log() const {
    return Vec256<double>(Sleef_logdx_u10sve(*this));
  }
  Vec256<double> log2() const {
    return Vec256<double>(Sleef_log2dx_u10sve(*this));
  }
  Vec256<double> log10() const {
    return Vec256<double>(Sleef_log10dx_u10sve(*this));
  }
  Vec256<double> log1p() const {
    return Vec256<double>(Sleef_log1pdx_u10sve(*this));
  }
  Vec256<double> frac() const;
  Vec256<double> sin() const {
    return Vec256<double>(Sleef_sindx_u10sve(*this));
  }
  Vec256<double> sinh() const {
    return Vec256<double>(Sleef_sinhdx_u10sve(*this));
  }
  Vec256<double> cos() const {
    return Vec256<double>(Sleef_cosdx_u10sve(*this));
  }
  Vec256<double> cosh() const {
    return Vec256<double>(Sleef_coshdx_u10sve(*this));
  }
  Vec256<double> ceil() const {
    return svrintp_f64_x(ptrue, *this);
  }
  Vec256<double> floor() const {
    return svrintm_f64_x(ptrue, *this);
  }
  Vec256<double> neg() const {
    return svneg_f64_x(ptrue, *this);
  }
  Vec256<double> round() const {
    return svrinti_f64_x(ptrue, *this);
  }
  Vec256<double> tan() const {
    return Vec256<double>(Sleef_tandx_u10sve(*this));
  }
  Vec256<double> tanh() const {
    return Vec256<double>(Sleef_tanhdx_u10sve(*this));
  }
  Vec256<double> trunc() const {
    return svrintz_f64_x(ptrue, *this);
  }
  Vec256<double> lgamma() const {
    return Vec256<double>(Sleef_lgammadx_u10sve(*this));
  }
  Vec256<double> sqrt() const {
    return svsqrt_f64_x(ptrue, *this);
  }
  Vec256<double> reciprocal() const {
    return svdivr_f64_x(ptrue, *this, svdup_n_f64(1.0));
  }
  Vec256<double> rsqrt() const {
    return svdivr_f64_x(ptrue, svsqrt_f64_x(ptrue, *this), svdup_n_f64(1.0));
  }
  Vec256<double> pow(const Vec256<double> &b) const {
    return Vec256<double>(Sleef_powdx_u10sve(*this, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<double> operator==(const Vec256<double>& other) const {
    svbool_t mask = svcmpeq_f64(ptrue, *this, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vec256<double> operator!=(const Vec256<double>& other) const {
    svbool_t mask = svcmpne_f64(ptrue, *this, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vec256<double> operator<(const Vec256<double>& other) const {
    svbool_t mask = svcmplt_f64(ptrue, *this, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vec256<double> operator<=(const Vec256<double>& other) const {
    svbool_t mask = svcmple_f64(ptrue, *this, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vec256<double> operator>(const Vec256<double>& other) const {
    svbool_t mask = svcmpgt_f64(ptrue, *this, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vec256<double> operator>=(const Vec256<double>& other) const {
    svbool_t mask = svcmpge_f64(ptrue, *this, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vec256<double> eq(const Vec256<double>& other) const;
  Vec256<double> ne(const Vec256<double>& other) const;
  Vec256<double> gt(const Vec256<double>& other) const;
  Vec256<double> ge(const Vec256<double>& other) const;
  Vec256<double> lt(const Vec256<double>& other) const;
  Vec256<double> le(const Vec256<double>& other) const;
};

template <>
Vec256<double> inline operator+(const Vec256<double>& a, const Vec256<double>& b) {
  return svadd_f64_x(ptrue, a, b);
}

template <>
Vec256<double> inline operator-(const Vec256<double>& a, const Vec256<double>& b) {
  return svsub_f64_x(ptrue, a, b);
}

template <>
Vec256<double> inline operator*(const Vec256<double>& a, const Vec256<double>& b) {
  return svmul_f64_x(ptrue, a, b);
}

template <>
Vec256<double> inline operator/(const Vec256<double>& a, const Vec256<double>& b) {
  return svdiv_f64_x(ptrue, a, b);
}

// frac. Implement this here so we can use subtraction
Vec256<double> Vec256<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<double> inline maximum(const Vec256<double>& a, const Vec256<double>& b) {
  return svmax_f64_x(ptrue, a, b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<double> inline minimum(const Vec256<double>& a, const Vec256<double>& b) {
  return svmin_f64_x(ptrue, a, b);
}

template <>
Vec256<double> inline clamp(const Vec256<double>& a, const Vec256<double>& min, const Vec256<double>& max) {
  return svminnm_f64_x(ptrue, max, svmaxnm_f64_x(ptrue, min, a));
}

template <>
Vec256<double> inline clamp_max(const Vec256<double>& a, const Vec256<double>& max) {
  return svminnm_f64_x(ptrue, max, a);
}

template <>
Vec256<double> inline clamp_min(const Vec256<double>& a, const Vec256<double>& min) {
  return svmaxnm_f64_x(ptrue, min, a);
}

template <>
Vec256<double> inline operator&(const Vec256<double>& a, const Vec256<double>& b) {
  return svreinterpret_f64_s64(svand_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

template <>
Vec256<double> inline operator|(const Vec256<double>& a, const Vec256<double>& b) {
  return svreinterpret_f64_s64(svorr_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

template <>
Vec256<double> inline operator^(const Vec256<double>& a, const Vec256<double>& b) {
  return svreinterpret_f64_s64(sveor_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

Vec256<double> Vec256<double>::eq(const Vec256<double>& other) const {
  return (*this == other) & Vec256<double>(1.0);
}

Vec256<double> Vec256<double>::ne(const Vec256<double>& other) const {
  return (*this != other) & Vec256<double>(1.0);
}

Vec256<double> Vec256<double>::gt(const Vec256<double>& other) const {
  return (*this > other) & Vec256<double>(1.0);
}

Vec256<double> Vec256<double>::ge(const Vec256<double>& other) const {
  return (*this >= other) & Vec256<double>(1.0);
}

Vec256<double> Vec256<double>::lt(const Vec256<double>& other) const {
  return (*this < other) & Vec256<double>(1.0);
}

Vec256<double> Vec256<double>::le(const Vec256<double>& other) const {
  return (*this <= other) & Vec256<double>(1.0);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
#pragma unroll
  for (int64_t i = 0; i < n; i += Vec256<double>::size()) {
    svbool_t pg = svwhilelt_b64(i, n);
    svst1_f64(pg, dst + i, svldnt1_f64(pg, src + i));
  }
}

template <>
Vec256<double> inline fmadd(const Vec256<double>& a, const Vec256<double>& b, const Vec256<double>& c) {
  return svmad_f64_x(ptrue, a, b, c);
}

#endif

}}}
