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
#include <ATen/native/ampl/activation.hpp>
#include <ATen/native/ampl/common.hpp>
#include <ATen/native/ampl/unary_ops.hpp>
#include <ATen/native/ampl/unary_ops_impl.hpp>

namespace ampl {

template <typename svbase_t>
inline svbase_t GeluForwardImpl(const svbool_t& pg, const svbase_t& x);

template <typename svbase_t>
inline svbase_t GeluBackwardImpl(
    const svbool_t& pg,
    const svbase_t& x,
    const svbase_t& dy);

#define AMPL_IMPLEMENT_GELU(bit)                                               \
  template <>                                                                  \
  inline svfloat##bit##_t GeluForwardImpl<svfloat##bit##_t>(                   \
      const svbool_t& pg, const svfloat##bit##_t& x) {                         \
    constexpr float##bit##_t alpha = M_SQRT1_2;                                \
    svfloat##bit##_t dst = Erf(pg, svmul_n_f##bit##_x(pg, x, alpha));          \
    return svmul_f##bit##_x(                                                   \
        pg, svmul_n_f##bit##_x(pg, svadd_n_f##bit##_x(pg, dst, 1.0), 0.5), x); \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline svfloat##bit##_t GeluBackwardImpl<svfloat##bit##_t>(                  \
      const svbool_t& pg,                                                      \
      const svfloat##bit##_t& x,                                               \
      const svfloat##bit##_t& dy) {                                            \
    constexpr float##bit##_t alpha = M_SQRT1_2;                                \
    constexpr float##bit##_t beta = M_2_SQRTPI * M_SQRT1_2 * 0.5;              \
                                                                               \
    svfloat##bit##_t cdf = Erf(pg, svmul_n_f##bit##_x(pg, x, alpha));          \
    cdf = svadd_n_f##bit##_x(pg, cdf, 1.0);                                    \
    cdf = svmul_n_f##bit##_x(pg, cdf, 0.5);                                    \
                                                                               \
    svfloat##bit##_t pdf =                                                     \
        Exp(pg, svmul_n_f##bit##_x(pg, svmul_f##bit##_x(pg, x, x), -0.5));     \
    pdf = svmul_n_f##bit##_x(pg, pdf, beta);                                   \
    return svmul_f##bit##_x(pg, dy, svmla_f##bit##_x(pg, cdf, x, pdf));        \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void GeluForward(                                                            \
      const int64_t n, const float##bit##_t* x, float##bit##_t* y) {           \
    parallel_for(                                                              \
        0,                                                                     \
        n,                                                                     \
        2048,                                                                  \
        GetVectorNumel<float##bit##_t>(),                                      \
        [&](int64_t begin, int64_t end) {                                      \
          for (int64_t i = begin; i < end;                                     \
               i += GetVectorNumel<float##bit##_t>()) {                        \
            const svbool_t pg = svwhilelt_b##bit(i, end);                      \
            const svfloat##bit##_t sv_x = svld1_f##bit(pg, x + i);             \
            svfloat##bit##_t dst = GeluForwardImpl(pg, sv_x);                  \
            svst1_f##bit(pg, y + i, dst);                                      \
          }                                                                    \
        });                                                                    \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void GeluBackward(                                                           \
      const int64_t n,                                                         \
      const float##bit##_t* x,                                                 \
      const float##bit##_t* dy,                                                \
      float##bit##_t* dx) {                                                    \
    parallel_for(                                                              \
        0,                                                                     \
        n,                                                                     \
        2048,                                                                  \
        GetVectorNumel<float##bit##_t>(),                                      \
        [&](int64_t begin, int64_t end) {                                      \
          for (int64_t i = begin; i < end;                                     \
               i += GetVectorNumel<float##bit##_t>()) {                        \
            const svbool_t pg = svwhilelt_b##bit(i, end);                      \
            const svfloat##bit##_t sv_x = svld1_f##bit(pg, x + i);             \
            const svfloat##bit##_t sv_dy = svld1_f##bit(pg, dy + i);           \
            svfloat##bit##_t dst = GeluBackwardImpl(pg, sv_x, sv_dy);          \
            svst1_f##bit(pg, dx + i, dst);                                     \
          }                                                                    \
        });                                                                    \
  }

AMPL_IMPLEMENT_GELU(16)
AMPL_IMPLEMENT_GELU(32)
AMPL_IMPLEMENT_GELU(64)

#undef AMPL_IMPLEMENT_GELU

} // namespace ampl
