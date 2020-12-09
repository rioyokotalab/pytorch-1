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
#ifndef AMPL_UNARY_OPS_IMPL_HPP_
#define AMPL_UNARY_OPS_IMPL_HPP_

#include <cmath>
#include <ATen/native/ampl/unary_ops.hpp>

namespace ampl {

#define AMPL_IMPLEMENT_CWISE_UOPS(bit)                                        \
  template <>                                                                 \
  inline svfloat##bit##_t Exp<svfloat##bit##_t>(                              \
      const svbool_t& pg, const svfloat##bit##_t& a) {                        \
    constexpr float##bit##_t cst_exp_hi = 88.3762626647950;                   \
    constexpr float##bit##_t cst_exp_lo = -88.3762626647949;                  \
    constexpr float##bit##_t cst_cephes_LOG2EF = 1.44269504088896341;         \
    constexpr float##bit##_t cst_nln2 = -0.6931471805599453;                  \
    const svfloat##bit##_t cst_exp_p0 = svdup_n_f##bit(1.9875691500e-4);      \
    const svfloat##bit##_t cst_exp_p1 = svdup_n_f##bit(1.3981999507e-3);      \
    const svfloat##bit##_t cst_exp_p2 = svdup_n_f##bit(8.3334519073e-3);      \
    const svfloat##bit##_t cst_exp_p3 = svdup_n_f##bit(4.1665795894e-2);      \
    const svfloat##bit##_t cst_exp_p4 = svdup_n_f##bit(1.6666665459e-1);      \
    const svfloat##bit##_t cst_exp_p5 = svdup_n_f##bit(5.0000001201e-1);      \
                                                                              \
    /* clamp */                                                               \
    const svfloat##bit##_t c = svmin_n_f##bit##_x(                            \
        pg, svmax_n_f##bit##_x(pg, a, cst_exp_lo), cst_exp_hi);               \
    const svfloat##bit##_t m =                                                \
        svrinti_f##bit##_x(pg, svmul_n_f##bit##_x(pg, a, cst_cephes_LOG2EF)); \
                                                                              \
    const svfloat##bit##_t r = svmla_n_f##bit##_x(pg, a, m, cst_nln2);        \
    const svfloat##bit##_t r2 = svmul_f##bit##_x(pg, r, r);                   \
    const svfloat##bit##_t r3 = svmul_f##bit##_x(pg, r2, r);                  \
                                                                              \
    svfloat##bit##_t y = svmla_f##bit##_x(pg, cst_exp_p1, cst_exp_p0, r);     \
    y = svmla_f##bit##_x(pg, cst_exp_p2, y, r);                               \
                                                                              \
    svfloat##bit##_t y1 = svmla_f##bit##_x(pg, cst_exp_p4, cst_exp_p3, r);    \
    y1 = svmla_f##bit##_x(pg, cst_exp_p5, y1, r);                             \
    y = svmla_f##bit##_x(pg, y1, y, r3);                                      \
                                                                              \
    const svfloat##bit##_t y2 = svadd_n_f##bit##_x(pg, r, 1.0);               \
    y = svmla_f##bit##_x(pg, y2, y, r2);                                      \
                                                                              \
    return svmax_f##bit##_x(                                                  \
        pg, svscale_f##bit##_x(pg, y, svcvt_s##bit##_f##bit##_x(pg, m)), a);  \
  }

AMPL_IMPLEMENT_CWISE_UOPS(16)
AMPL_IMPLEMENT_CWISE_UOPS(32)
AMPL_IMPLEMENT_CWISE_UOPS(64)

#undef AMPL_IMPLEMENT_CWISE_UOPS

#define AMPL_IMPLEMENT_CWISE_UOPS(bit)                                       \
  template <>                                                                \
  inline svfloat##bit##_t Erf<svfloat##bit##_t>(                             \
      const svbool_t& pg, const svfloat##bit##_t& a) {                       \
    const svfloat##bit##_t alpha_1 = svdup_n_f##bit(-1.60960333262415e-02);  \
    const svfloat##bit##_t alpha_3 = svdup_n_f##bit(-2.95459980854025e-03);  \
    const svfloat##bit##_t alpha_5 = svdup_n_f##bit(-7.34990630326855e-04);  \
    const svfloat##bit##_t alpha_7 = svdup_n_f##bit(-5.69250639462346e-05);  \
    const svfloat##bit##_t alpha_9 = svdup_n_f##bit(-2.10102402082508e-06);  \
    const svfloat##bit##_t alpha_11 = svdup_n_f##bit(2.77068142495902e-08);  \
    const svfloat##bit##_t alpha_13 = svdup_n_f##bit(-2.72614225801306e-10); \
    const svfloat##bit##_t beta_0 = svdup_n_f##bit(-1.42647390514189e-02);   \
    const svfloat##bit##_t beta_2 = svdup_n_f##bit(-7.37332916720468e-03);   \
    const svfloat##bit##_t beta_4 = svdup_n_f##bit(-1.68282697438203e-03);   \
    const svfloat##bit##_t beta_6 = svdup_n_f##bit(-2.13374055278905e-04);   \
    const svfloat##bit##_t beta_8 = svdup_n_f##bit(-1.45660718464996e-05);   \
                                                                             \
    /* clamp */                                                              \
    const svfloat##bit##_t c =                                               \
        svmin_n_f##bit##_x(pg, svmax_n_f##bit##_x(pg, a, -4.0), 4.0);        \
    const svfloat##bit##_t c2 = svmul_f##bit##_x(pg, c, c);                  \
                                                                             \
    svfloat##bit##_t p = svmla_f##bit##_x(pg, alpha_11, c2, alpha_13);       \
    p = svmla_f##bit##_x(pg, alpha_9, c2, p);                                \
    p = svmla_f##bit##_x(pg, alpha_7, c2, p);                                \
    p = svmla_f##bit##_x(pg, alpha_5, c2, p);                                \
    p = svmla_f##bit##_x(pg, alpha_3, c2, p);                                \
    p = svmla_f##bit##_x(pg, alpha_1, c2, p);                                \
    p = svmul_f##bit##_x(pg, c, p);                                          \
                                                                             \
    svfloat##bit##_t q = svmla_f##bit##_x(pg, beta_6, c2, beta_8);           \
    q = svmla_f##bit##_x(pg, beta_4, c2, q);                                 \
    q = svmla_f##bit##_x(pg, beta_2, c2, q);                                 \
    q = svmla_f##bit##_x(pg, beta_0, c2, q);                                 \
                                                                             \
    return svdiv_f##bit##_x(pg, p, q);                                       \
  }

AMPL_IMPLEMENT_CWISE_UOPS(16)
AMPL_IMPLEMENT_CWISE_UOPS(32)
AMPL_IMPLEMENT_CWISE_UOPS(64)

} // namespace ampl

#undef AMPL_IMPLEMENT_CWISE_UOPS

#endif // AMPL_UNARY_OPS_IMPL_HPP_
