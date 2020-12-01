/*
 Copyright (c) 2020, RIKEN, Japan
 Copyright (c) 2020, FUJITSU LIMITED
 All rights reserved.

 Written by Kazutoshi Akao(FUJITSU LIMITED).

 This code was created based on the following Research Poster of ISC18.

 Automatic Generation of Full-Set Batched BLAS
 Yusuke Hirota, Daichi Mukunoki, Toshiyuki Imamura (RIKEN, Japan)
 Research Poster, International Supercomputing Conference (ISC'18) 26 June 2018

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

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/cpu/LinearAlgebra.h>

#include <TH/THBlasUtils.h>

#if !AT_BUILD_WITH_BLAS()

namespace at { namespace native {
namespace {

Tensor& _baddbmm_blas_(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  AT_ERROR("bmm: ATen not compiled with BLAS support");
}

} // namespace

REGISTER_DISPATCH(baddbmm_blas_stub, &_baddbmm_blas_);

}}

#else // AT_BUILD_WITH_BLAS

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

namespace {

double get_cost_side_n3(const std::vector<int>& cost_param) {
  int n1, n2;

  n1 = cost_param[1];
  n2 = cost_param[2];

  return (double)n1 * (double)n2 * (double)n2;
}

static int pick_min_thread(const double *cost, const int n) {
  int current_min_i = 0;
  for (int i = 1; i < n; ++i) {
    if (cost[i] < cost[current_min_i]) {
      current_min_i = i;
    }
  }
  return current_min_i;
}

void schedule_batch(int batch_size, double (*cost_func)(const std::vector<int>& cost_param),
		    const std::vector<int>& cost_param, std::vector<int>& which_thread) {
  const int num_threads = at::get_num_threads();
  double cost;
  double current_cost[num_threads]; // current_cost[i] = j : Current cost of the i-th thread is j.

  // Greedy task allocation
  for (int tid = 0; tid < num_threads; ++tid) {
    current_cost[tid] = 0.0;
  }

  cost = cost_func(cost_param);

  // Task allocation
  for (int local_no = 0; local_no < batch_size; ++local_no) {
    const int min_thread = pick_min_thread(current_cost, num_threads);
    which_thread[local_no] = min_thread;
    current_cost[which_thread[local_no]] += cost;
  }
}

// condition for using batch
//  return 1:use , 0:not use
int use_batch() {
  const int num_threads = at::get_num_threads();

  if(num_threads > 1)
    return 1;
  else
    return 0;
}

} // namespace

namespace at { namespace native {
namespace {

template <typename scalar_t>
static inline void baddbmm_blas_template(const Tensor& res, const Tensor& mat1, const Tensor& mat2, Scalar beta_, Scalar alpha_) {
  auto is_transposed = [&](const TensorAccessor<scalar_t, 2>& t) {
    return t.stride(0) == 1 && t.stride(1) >= t.size(0);
  };

  auto mat1_acc = mat1.accessor<scalar_t, 3>();
  auto mat2_acc = mat2.accessor<scalar_t, 3>();
  auto res_acc = res.accessor<scalar_t, 3>();

  const char trans_A = is_transposed(mat1_acc[0]) ? 't' : 'n';
  const char trans_B = is_transposed(mat2_acc[0]) ? 't' : 'n';

  const int batch_size = mat1_acc.size(0);
  const int M = mat1_acc.size(1);
  const int N = mat2_acc.size(2);
  const int K = mat1_acc.size(2);
  scalar_t alpha = alpha_.to<scalar_t>();
  scalar_t beta = beta_.to<scalar_t>();

  const int lda = is_transposed(mat1_acc[0]) ? mat1_acc[0].stride(1) : mat1_acc[0].stride(0);
  const int ldb = is_transposed(mat2_acc[0]) ? mat2_acc[0].stride(1) : mat2_acc[0].stride(0);
  const int ldc = res[0].stride(0);

  std::vector<scalar_t*> A(batch_size);
  std::vector<scalar_t*> B(batch_size);
  std::vector<scalar_t*> C(batch_size);

  for (int64_t batch = 0; batch < batch_size; batch++) {
    A[batch] = mat1_acc[batch].data();
    B[batch] = mat2_acc[batch].data();
    C[batch] = res_acc[batch].data();
  }

  if (!use_batch()) {
    for (int local_no = 0; local_no < batch_size; ++local_no) {
      THBlas_gemm<scalar_t>(trans_B, trans_A, N, M, K, alpha, B[local_no], ldb,
			    A[local_no], lda, beta, C[local_no], ldc);
    }
    return;
  }

  std::vector<int> which_thread(batch_size);
  std::vector<int> cost_param{M, N, K};
  schedule_batch(batch_size, get_cost_side_n3, cost_param, which_thread);

  at::parallel_for(0, at::get_num_threads(), 0, [&](int64_t start, int64_t end) {
    int my_tno = at::get_thread_num();
    for (int local_no = 0; local_no < batch_size; ++local_no) {
      if(which_thread[local_no] == my_tno){
        THBlas_gemm<scalar_t>(trans_B, trans_A, N, M, K, alpha, B[local_no], ldb,
			      A[local_no], lda, beta, C[local_no], ldc);
      }
    }
  });
}

Tensor& _baddbmm_blas_(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  // checks are done in native/LinearAlgebra.cpp
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "baddbmm__blas", [&] {
      baddbmm_blas_template<scalar_t>(self, batch1, batch2, beta, alpha);
    });

  return self;
}

} // namespace

REGISTER_DISPATCH(baddbmm_blas_stub, &_baddbmm_blas_);

}} // namespace at::native

#endif // USE_BLAS
