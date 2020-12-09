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
#ifndef AMPL_COMMON_HPP_
#define AMPL_COMMON_HPP_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ATen/native/ampl/ampl.h>

namespace ampl {

// Count the number of N-bit elements from Standard dtype
template <typename T>
inline int64_t GetVectorNumel();

#define AMPL_IMPLEMENT_CNT(type, cnttype) \
  template <>                             \
  inline int64_t GetVectorNumel<type>() { \
    return svcnt##cnttype();              \
  }

AMPL_IMPLEMENT_CNT(float16_t, h)
AMPL_IMPLEMENT_CNT(float32_t, w)
AMPL_IMPLEMENT_CNT(float64_t, d)

#undef AMPL_IMPLEMENT_CNT

/**
* Do loop parallel for
*
* @param[in] begin: index at which to start applying user function
* @param[in] end: index at which to stop applying user function
* @param[in] grain_size: minimum number of elements per chunk
* @param[in] vector_numel: number of vector elements used for chunk size aligment
* @param[in] f: functions
*/
template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const int64_t vector_numel,
    const F& f) {
  if (begin >= end)
    return;
#ifdef _OPENMP
  const int64_t N = end - begin;
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  // Work around memory leak when using 1 thread in nested "omp parallel"
  // caused by some buggy OpenMP versions and the fact that omp_in_parallel()
  // returns false when omp_get_max_threads() == 1 inside nested "omp parallel"
  // See issue gh-32284

#pragma omp parallel if ( \
    omp_get_max_threads() > 1 && !omp_in_parallel() && (N > grain_size))
  {
    // choose number of tasks based on grain size and number of threads
    // can't use num_threads clause due to bugs in GOMP's thread pool (See
    // #32008)
    const int64_t num_threads = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t chunk_size = (N + num_threads - 1) / num_threads;
    const int64_t chunk_size_aligned =
        ((chunk_size + vector_numel - 1) / vector_numel) * vector_numel;
    const int64_t begin_tid = begin + tid * chunk_size_aligned;
    if (begin_tid < end) {
      try {
        f(begin_tid, std::min(end, chunk_size_aligned + begin_tid));
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
#else
  f(begin, end);
#endif
}

} // namespace ampl

#endif // AMPL_COMMON_HPP_
