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
#ifndef AMPL_UNARY_OPS_HPP_
#define AMPL_UNARY_OPS_HPP_
#include <ATen/native/ampl/ampl.h>

namespace ampl {

/**
 * Computes an exponential of SVE data type elements.
 *
 * @param[in] pg Predicate for bits control.
 * @param[in] a Input SVE data type vector.
 * @return Exp(a)
 * @attention Support types are svfloat16_t, svfloat32_t, svfloat64_t
 */
template <typename svbase_t>
inline svbase_t Exp(const svbool_t& pg, const svbase_t& a);

/**
 * Computes an exponential of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] a Pointer to an array that contains the input vector a.
 * @param[out] y Pointer to an array that contains the output vector y.
 * @attention Support types are float16_t, float32_t, float64_t
 */
template <typename scalar_t>
void vExp(const int64_t n, const scalar_t* a, scalar_t* y);

/**
 * Computes an error function of SVE data type elements.
 *
 * @param[in] pg Predicate for bits control.
 * @param[in] a Input SVE data type vector.
 * @return Erf(a)
 * @attention Support types are svfloat16_t, svfloat32_t, svfloat64_t
 */
template <typename svbase_t>
inline svbase_t Erf(const svbool_t& pg, const svbase_t& a);

/**
 * Computes the error function value of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] a Pointer to an array that contains the input vector a.
 * @param[out] y Pointer to an array that contains the output vector y.
 * @attention Support types are float16_t, float32_t, float64_t
 */
template <typename scalar_t>
void vErf(const int64_t n, const scalar_t* a, scalar_t* y);

} // namespace ampl

#endif // AMPL_UNARY_OPS_HPP_
