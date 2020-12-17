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

/**
 * C-API for AMPL
 * User-level math functions and statistical functions using SVE.
 * API specifications follow MKL-VML and MKL-VSL
 *
 */

#ifdef __ARM_FEATURE_SVE
#ifndef AMPL_AMPL_H_
#define AMPL_AMPL_H_

#include <arm_sve.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * StreamStatePtr stream for AMPL Random function
 *
 */
struct StreamStatePtr_ampl {
  int METHOD; /* METHOD 0:MCG31 */
  uint32_t* s; /* seed */
};


/**
 * Preparation for sve calculation.
 *
 * @param[out] stream
 * @param[in] METHOD 0:MCG31
 * @param[in] seed The initial state for random number generation.
 */
void NewStream_ampl(struct StreamStatePtr_ampl* stream, int METHOD, int seed);

/**
 * To support multi-threaded computation.
 *
 * @param[in, out] StreamStatePtr_ampl stream
 * @param[in] begin Number of jumping ahead.
 */
void SkipAheadStream_ampl(struct StreamStatePtr_ampl stream, int begin);

/**
 * Generation of random numbers by Bernoulli distribution.
 *
 * @param[out] r Value (0 or 1) genereted by the Bernoulli function.
 * @param[in] METHOD 0:MCG31
 * @param[in] StreamStatePtr_ampl stream
 * @param[in] N Number of random numbers to generate.
 * @param[in] r Array of random numbers.
 * @param[in] p Parameters of the Bernoulli distribution.
 * @return 0
 */
int RngBernoulli_ampl(
    int METHOD,
    struct StreamStatePtr_ampl stream,
    int N,
    int* r,
    double p);

/**
 * Delete stream.
 *
 * @param[in] StreamStatePtr_ampl stream
 */
void DeleteStream_ampl(struct StreamStatePtr_ampl stream);

/**
 * Computes an exponential of SVE data type elements.
 *
 * @param[in] pg Predicate for bits control.
 * @param[in] a Input SVE data type vector.
 * @return Exp(a)
 * @attention Support types are svfloat16_t, svfloat32_t, svfloat64_t
 */
svfloat16_t hExp_ampl(const svbool_t pg, const svfloat16_t a);
svfloat32_t sExp_ampl(const svbool_t pg, const svfloat32_t a);
svfloat64_t dExp_ampl(const svbool_t pg, const svfloat64_t a);

/**
 * Computes an exponential of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] a Pointer to an array that contains the input vector a.
 * @param[out] y Pointer to an array that contains the output vector y.
 * @attention Support types are float16_t, float32_t, float64_t
 */
void vhExp_ampl(const int64_t n, const float16_t* a, float16_t* y);
void vsExp_ampl(const int64_t n, const float32_t* a, float32_t* y);
void vdExp_ampl(const int64_t n, const float64_t* a, float64_t* y);

/**
 * Computes an error function of SVE data type elements.
 *
 * @param[in] pg Predicate for bits control.
 * @param[in] a Input SVE data type vector.
 * @return Erf(a)
 * @attention Support types are svfloat16_t, svfloat32_t, svfloat64_t
 */
svfloat16_t hErf_ampl(const svbool_t pg, const svfloat16_t a);
svfloat32_t sErf_ampl(const svbool_t pg, const svfloat32_t a);
svfloat64_t dErf_ampl(const svbool_t pg, const svfloat64_t a);

/**
 * Computes the error function value of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] a Pointer to an array that contains the input vector a.
 * @param[out] y Pointer to an array that contains the output vector y.
 * @attention Support types are float16_t, float32_t, float64_t
 */
void vhErf_ampl(const int64_t n, const float16_t* a, float16_t* y);
void vsErf_ampl(const int64_t n, const float32_t* a, float32_t* y);
void vdErf_ampl(const int64_t n, const float64_t* a, float64_t* y);

#ifdef __cplusplus
} // extern C
#endif // __cplusplus

#endif // AMPL_AMPL_H_
#endif // __ARM_FEATURE_SVE
