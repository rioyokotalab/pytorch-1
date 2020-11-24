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

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#include <ATen/native/quantized/affine_quantizer.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint8.h>

#include <array>

// This file defines Vec256<> for the quantized types.
//
//
// Currently, we simply use these classes as efficient converters between
// the quantized types and Vec256<float>, usually in bandwidth-bound cases
// where doing the arithmetic in full-precision is acceptable (e.g.
// elementwise operators).
//
//
// Conversions are as follows:
//  Vec256<qint8> -> 4x Vec256<float>
//  Vec256<quint8> -> 4x Vec256<float>
//  Vec256<qint32> -> 1x Vec256<float>
//
// The size of the returned float vector is specified by the special
// constexpr function float_num_vecs. The type of the value returned
// from dequantize (and expected as an argument to quantize) is
// specified by float_vec_return_type.
//
// When writing kernels with these vectors, it is expected that floating-
// point operations will be carried out in a loop over Vec256<T>::float_num_vecs
// iterations.

namespace at {
namespace vec256 {
namespace {

#if defined(__GNUC__) && defined(__ARM_FEATURE_SVE)

// NOTE: These are low-performance implementations that we fall back on
// if we are not building with AVX2. This may not be an issue, because
// currently for quantization we assume the user has at least AVX512
// installed, so these can simply act as a reference implementation.
//
// If in the future we relax this requirement (AVX2+), we should probably
// revisit these implementations

template <
    typename T,
    typename float_vec_return_type_,
    typename int_vec_return_type_,
    int size_>
struct Vec256QuantizedConverter {
  static constexpr int size() {
    return size_;
  }

  static constexpr int float_num_vecs() {
    return size() / Vec256<float>::size();
  }

  static constexpr int int_num_vecs() {
    return size() / Vec256<int32_t>::size();
  }

  using float_vec_return_type = float_vec_return_type_;
  using int_vec_return_type = int_vec_return_type_;

  using value_type = typename T::underlying;
  std::array<value_type, size_> vals;

  Vec256QuantizedConverter(T val) {
    for (size_t i = 0; i < size(); ++i) {
      vals[i] = val.val_;
    }
  }

  Vec256QuantizedConverter(const void* ptr) {
    memcpy(vals.data(), ptr, sizeof(value_type) * size());
  }

  void store(void* ptr, int count = size()) const {
    memcpy(ptr, vals.data(), count * sizeof(value_type));
  }

  float_vec_return_type dequantize(
      Vec256<float> scale,
      Vec256<float> zero_point,
      Vec256<float> scale_zp_premul) const {
    float_vec_return_type rv;
    float tmp_scale[Vec256<float>::size()];
    float tmp_zero_point[Vec256<float>::size()];
    scale.store(tmp_scale);
    zero_point.store(tmp_zero_point);
    for (int i = 0; i < float_num_vecs(); ++i) {
      float tmp_vals[Vec256<float>::size()];
      for (int j = 0; j < Vec256<float>::size(); ++j) {
        tmp_vals[j] =
          at::native::dequantize_val<T>(tmp_scale[j], tmp_zero_point[j], T(vals[Vec256<float>::size() * i + j]));
      }
      rv[i] = Vec256<float>::loadu(tmp_vals);
    }
    return rv;
  }

  void dump() const {
      for (int i = 0; i < size(); ++i) {
          std::cout << vals[i] << " ";
      }
      std::cout << std::endl;
  }

 protected:
  Vec256QuantizedConverter() {}
};

template <>
struct Vec256<c10::qint32> : public Vec256QuantizedConverter<
                                 c10::qint32,
                                 std::array<Vec256<float>, 1>,
                                 std::array<Vec256<c10::qint32>, 1>,
                                 VECTOR_BYTE_SIZE / 4> {
  Vec256()
      : Vec256QuantizedConverter<
            c10::qint32,
            std::array<Vec256<float>, 1>,
            std::array<Vec256<c10::qint32>, 1>,
            VECTOR_BYTE_SIZE / 4>() {}
  Vec256(c10::qint32 val)
      : Vec256QuantizedConverter<
            c10::qint32,
            std::array<Vec256<float>, 1>,
            std::array<Vec256<c10::qint32>, 1>,
            VECTOR_BYTE_SIZE / 4>(val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<
            c10::qint32,
            std::array<Vec256<float>, 1>,
            std::array<Vec256<c10::qint32>, 1>,
            VECTOR_BYTE_SIZE / 4>(ptr) {}

  static Vec256<c10::qint32> loadu(const void* ptr) {
    return Vec256<c10::qint32>(ptr);
  }

  static Vec256<c10::qint32> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * Vec256<float>::size()> float_vals;

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(&float_vals[i * Vec256<float>::size()], Vec256<float>::size());
    }

    at::native::quantize_vec<c10::qint32, /*precision=*/32>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint32*)qvals.data(),
        Vec256<float>::size() * float_num_vecs());

    return Vec256<c10::qint32>::loadu(qvals.data());
  }

  Vec256<c10::qint32> maximum(Vec256<c10::qint32> b) const {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint32> minimum(Vec256<c10::qint32> b) const {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint32> relu(Vec256<c10::qint32> zero_point) const  {
    return maximum(zero_point);
  }


  Vec256<c10::qint32> relu6(
      Vec256<c10::qint32> zero_point,
      Vec256<c10::qint32> q_six) {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vec256<c10::qint32> b) const {
    int_vec_return_type retval;
    for (size_t i = 0; i < size(); ++i) {
      retval[0].vals[i] = vals[i] - b.vals[i];
    }
    return retval;
  }

  static Vec256<c10::qint32> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    Vec256<c10::qint32> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] =
          nearbyint(static_cast<float>(inp[0].vals[i]) * multiplier) +
          zero_point;
    }
    return retval;
  }
};

template <>
Vec256<c10::qint32> inline maximum(const Vec256<c10::qint32>& a, const Vec256<c10::qint32>& b) {
  return a.maximum(b);
}

template <>
Vec256<c10::qint32> inline operator*(
    const Vec256<c10::qint32>& a,
    const Vec256<c10::qint32>& b) {
  Vec256<c10::qint32> retval;
  for (size_t i = 0; i < std::decay_t<decltype(a)>::size(); ++i) {
    retval.vals[i] = a.vals[i] * b.vals[i];
  }
  return retval;
}

template <>
Vec256<c10::qint32> inline operator+(
    const Vec256<c10::qint32>& a,
    const Vec256<c10::qint32>& b) {
  Vec256<c10::qint32> retval;
  for (size_t i = 0; i < std::decay_t<decltype(a)>::size(); ++i) {
    retval.vals[i] = a.vals[i] + b.vals[i];
  }
  return retval;
}

template <>
struct Vec256<c10::qint8> : public Vec256QuantizedConverter<
                                c10::qint8,
                                std::array<Vec256<float>, 4>,
                                std::array<Vec256<c10::qint32>, 4>,
                                VECTOR_BYTE_SIZE> {
  Vec256()
      : Vec256QuantizedConverter<
            c10::qint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            VECTOR_BYTE_SIZE>() {}
  Vec256(c10::qint8 val)
      : Vec256QuantizedConverter<
            c10::qint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            VECTOR_BYTE_SIZE>(val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<
            c10::qint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            VECTOR_BYTE_SIZE>(ptr) {}

  static Vec256<c10::qint8> loadu(const void* ptr) {
    return Vec256<c10::qint8>(ptr);
  }

  static Vec256<c10::qint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * Vec256<float>::size()> float_vals;

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(&float_vals[i * Vec256<float>::size()], Vec256<float>::size());
    }

    at::native::quantize_vec<c10::qint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::qint8*)qvals.data(),
        Vec256<float>::size() * float_num_vecs());

    return Vec256<c10::qint8>::loadu(qvals.data());
  }

  Vec256<c10::qint8> maximum(Vec256<c10::qint8> b) const {
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint8> minimum(Vec256<c10::qint8> b) const {
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::qint8> relu(Vec256<c10::qint8> zero_point) const {
    return maximum(zero_point);
  }

  Vec256<c10::qint8> relu6(
      Vec256<c10::qint8> zero_point,
      Vec256<c10::qint8> q_six) {
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vec256<c10::qint8> b) const {
    int_vec_return_type retval;
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    for (size_t i = 0; i < int_num_vecs(); ++i) {
      for (size_t j = 0; j < elem_per_int_vec; ++j) {
        retval[i].vals[j] =
            static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
            static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
      }
    }
    return retval;
  }
  static Vec256<c10::qint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    constexpr auto min_val = std::numeric_limits<value_type>::min();
    constexpr auto max_val = std::numeric_limits<value_type>::max();
    Vec256<c10::qint8> retval;
    for (size_t i = 0; i < int_num_vecs(); ++i) {
      for (size_t j = 0; j < elem_per_int_vec; ++j) {
        int32_t rounded =
            nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
            zero_point;
        retval.vals[i * elem_per_int_vec + j] =
            std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
      }
    }
    return retval;
  }
};

template <>
Vec256<c10::qint8> inline maximum(const Vec256<c10::qint8>& a, const Vec256<c10::qint8>& b) {
  return a.maximum(b);
}

template <>
struct Vec256<c10::quint8> : public Vec256QuantizedConverter<
                                 c10::quint8,
                                 std::array<Vec256<float>, 4>,
                                 std::array<Vec256<c10::qint32>, 4>,
                                 VECTOR_BYTE_SIZE> {
  Vec256()
      : Vec256QuantizedConverter<
            c10::quint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            VECTOR_BYTE_SIZE>() {}
  Vec256(c10::quint8 val)
      : Vec256QuantizedConverter<
            c10::quint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            VECTOR_BYTE_SIZE>(val) {}
  Vec256(const void* ptr)
      : Vec256QuantizedConverter<
            c10::quint8,
            std::array<Vec256<float>, 4>,
            std::array<Vec256<c10::qint32>, 4>,
            VECTOR_BYTE_SIZE>(ptr) {}

  static Vec256<c10::quint8> loadu(const void* ptr) {
    return Vec256<c10::quint8>(ptr);
  }

  static Vec256<c10::quint8> quantize(
      const float_vec_return_type& rhs,
      float scale,
      int32_t zero_point,
      float inverse_scale) {
    std::array<value_type, size()> qvals;
    std::array<float, float_num_vecs() * Vec256<float>::size()> float_vals;

    for (int i = 0; i < float_num_vecs(); ++i) {
      rhs[i].store(&float_vals[i * Vec256<float>::size()], Vec256<float>::size());
    }

    at::native::quantize_vec<c10::quint8>(
        scale,
        zero_point,
        float_vals.data(),
        (c10::quint8*)qvals.data(),
        Vec256<float>::size() * float_num_vecs());

    return Vec256<c10::quint8>::loadu(qvals.data());
  }

  Vec256<c10::quint8> maximum(Vec256<c10::quint8> b) const {
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::max<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::quint8> minimum(Vec256<c10::quint8> b) const {
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(vals[i], b.vals[i]);
    }
    return retval;
  }

  Vec256<c10::quint8> relu(Vec256<c10::quint8> zero_point) const {
    return maximum(zero_point);
  }


  Vec256<c10::quint8> relu6(
      Vec256<c10::quint8> zero_point,
      Vec256<c10::quint8> q_six) {
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < size(); ++i) {
      retval.vals[i] = std::min<value_type>(
          std::max<value_type>(vals[i], zero_point.vals[i]), q_six.vals[i]);
    }
    return retval;
  }

  int_vec_return_type widening_subtract(Vec256<c10::quint8> b) const {
    int_vec_return_type retval;
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    for (size_t i = 0; i < int_num_vecs(); ++i) {
      for (size_t j = 0; j < elem_per_int_vec; ++j) {
        retval[i].vals[j] =
            static_cast<int32_t>(vals[i * elem_per_int_vec + j]) -
            static_cast<int32_t>(b.vals[i * elem_per_int_vec + j]);
      }
    }
    return retval;
  }
  static Vec256<c10::quint8> requantize_from_int(
      const int_vec_return_type& inp,
      float multiplier,
      int32_t zero_point) {
    constexpr int elem_per_int_vec = size() / int_num_vecs();
    constexpr auto min_val = std::numeric_limits<value_type>::min();
    constexpr auto max_val = std::numeric_limits<value_type>::max();
    Vec256<c10::quint8> retval;
    for (size_t i = 0; i < int_num_vecs(); ++i) {
      for (size_t j = 0; j < elem_per_int_vec; ++j) {
        int32_t rounded =
            nearbyint(static_cast<float>(inp[i].vals[j]) * multiplier) +
            zero_point;
        retval.vals[i * elem_per_int_vec + j] =
            std::min<int32_t>(std::max<int32_t>(rounded, min_val), max_val);
      }
    }
    return retval;
  }
};

template <>
Vec256<c10::quint8> inline maximum(const Vec256<c10::quint8>& a, const Vec256<c10::quint8>& b) {
  return a.maximum(b);
}

#endif // defined(__GNUC__) && defined(__ARM_FEATURE_SVE)

}}}
