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
#ifndef AMPL_ACTIVATION_HPP_
#define AMPL_ACTIVATION_HPP_

#include <cmath>
#include <ATen/native/ampl/ampl.h>

namespace ampl {

/**
 * Computes the GELU forward value of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] x Pointer to an array that contains the input vector x.
 * @param[out] y Pointer to an array that contains the output vector y.
 */
template <typename scalar_t>
void GeluForward(
    const int64_t n,
    const scalar_t* x,
    scalar_t* y);

/**
 * Computes the GELU backward value of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] x Pointer to an array that contains the input vector x.
 * @param[in] dy Pointer to an array that contains the input vector dy.
 * @param[out] dx Pointer to an array that contains the output vector dx.
 */
template <typename scalar_t>
void GeluBackward(
    const int64_t n,
    const scalar_t* x,
    const scalar_t* dy,
    scalar_t* dx);

} // namespace ampl

#endif // AMPL_ACTIVATION_HPP_
