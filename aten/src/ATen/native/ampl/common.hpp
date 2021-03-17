#ifdef __ARM_FEATURE_SVE

#ifndef AMPL_COMMON_HPP_
#define AMPL_COMMON_HPP_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>

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
#endif // __ARM_FEATURE_SVE
