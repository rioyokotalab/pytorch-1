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

#include <ATen/native/cpu/Loss.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#include <ATen/native/ampl/ampl.hpp>
#endif

#if !defined(__ARM_FEATURE_SVE)

namespace at {
namespace native {
namespace {

Tensor _kl_div_backward(const Tensor & grad, const Tensor & input,
			const Tensor & target, const int64_t reduction,
			const bool log_target) {
  AT_ERROR("_kl_div_backward: This CPU is not supported yet");
}

} // namespace
} // namespace native
} // namespace at

#endif

namespace at {
namespace native {
namespace {

#ifdef __ARM_FEATURE_SVE

#define ptrue svptrue_b8()

#define SVE_KL_DIV_BACKWARD_IMPL_TEMPLATE(type, bit, svcnt_func)                           \
static inline void kl_div_backward_impl(type * _grad_input, const int64_t grad_input_size, \
                                        const type * _grad, const int64_t grad_size,       \
                                        const type * _target, const int64_t target_size,   \
                                        const int64_t reduction, const bool log_target) {  \
  svfloat##bit##_t const_grad, const_target;						   \
  svfloat##bit##_t zero = svdup_n_f##bit((type)0);                                         \
  if (grad_size == 1) const_grad = svdup_n_f##bit(*_grad);                                 \
  if (target_size == 1) const_target = svdup_n_f##bit(*_target);                           \
                                                                                           \
  if (!log_target) {							                   \
    at::parallel_for(0, grad_input_size, 2048, [&](int64_t start, int64_t end) {           \
      int64_t size = end - start;                                                          \
      int64_t offset;                                                                      \
      svbool_t pg, mask;							           \
      svfloat##bit##_t grad, target, grad_input;				           \
                                                                                           \
      for (int64_t i = 0; i < size; i += svcnt_func()) {                                   \
	offset = start + i;                                                                \
	pg = svwhilelt_b##bit(i, size);                                                    \
	grad = grad_size == 1 ? const_grad : svld1_f##bit(pg, _grad + offset);             \
	target = target_size == 1 ? const_target : svld1_f##bit(pg, _target + offset);     \
	mask = svcmpgt_f##bit(pg, target, zero);                                           \
	grad_input = svmul_f##bit##_z(mask, svneg_f##bit##_x(ptrue, target), grad);        \
	svst1_f##bit(pg, _grad_input + offset, grad_input);                                \
      }                                                                                    \
    });                                                                                    \
  } else {                                                                                 \
    at::parallel_for(0, grad_input_size, 2048, [&](int64_t start, int64_t end) {           \
      int64_t size = end - start;                                                          \
      int64_t offset;                                                                      \
      svbool_t pg;                                                                         \
      svfloat##bit##_t grad, target, grad_input;            			           \
                                                                                           \
      for (int64_t i = 0; i < size; i += svcnt_func()) {                                   \
        offset = start + i;                                                                \
        pg = svwhilelt_b##bit(i, size);                                                    \
        grad = grad_size == 1 ? const_grad : svld1_f##bit(pg, _grad + offset);             \
        target = target_size == 1 ? const_target : svld1_f##bit(pg, _target + offset);     \
        grad_input = svmul_f##bit##_x(ptrue,                                               \
				      svneg_f##bit##_x(ptrue, ampl::Exp(ptrue, target)),   \
				      grad);				                   \
        svst1_f##bit(pg, _grad_input + offset, grad_input);                                \
      }                                                                                    \
    });                                                                                    \
  }                                                                                        \
                                                                                           \
  if (reduction == at::Reduction::Mean) {                                                  \
    svfloat##bit##_t recp_numel = svdup_n_f##bit(1 / (type)grad_input_size);               \
    at::parallel_for(0, grad_input_size, 2048, [&](int64_t start, int64_t end) {           \
      int64_t size = end - start;                                                          \
      int64_t offset;                                                                      \
      svbool_t pg;                                                                         \
      svfloat##bit##_t grad_input;                                                         \
                                                                                           \
      for (int64_t i = 0; i < size; i += svcnt_func()) {                                   \
	offset = start + i;                                                                \
	pg = svwhilelt_b##bit(i, size);                                                    \
	grad_input = svld1_f##bit(pg, _grad_input + offset);                               \
	grad_input = svmul_f##bit##_x(pg, grad_input, recp_numel);                         \
	svst1_f##bit(pg, _grad_input + offset, grad_input);                                \
      }                                                                                    \
    });                                                                                    \
  }                                                                                        \
}

SVE_KL_DIV_BACKWARD_IMPL_TEMPLATE(float, 32, svcntw)
SVE_KL_DIV_BACKWARD_IMPL_TEMPLATE(double, 64, svcntd)

template <typename scalar_t>
static inline void kl_div_backward_template(const Tensor& grad_input, const Tensor& _grad,
					    const Tensor& _target, const int64_t reduction,
					    const bool log_target) {
  int64_t grad_size, target_size;
  for (int64_t i = 0; i < _grad.dim(); ++i) {
    if (_grad.strides()[i] == 0) {
      grad_size = 1;
    } else {
      grad_size = _grad.numel();
      break;
    }
  }
  for (int64_t i = 0; i < _target.dim(); ++i) {
    if (_target.strides()[i] == 0) {
      target_size = 1;
    } else {
      target_size = _target.numel();
      break;
    }
  }

  Tensor grad = _grad;
  if (grad_size != 1 && !_grad.is_contiguous()) {
    grad = _grad.contiguous();
  }
  Tensor target = _target;
  if (target_size != 1 && !_target.is_contiguous()) {
    target = _target.contiguous();
  }

  kl_div_backward_impl(grad_input.data_ptr<scalar_t>(), grad_input.numel(),
		       grad.data_ptr<scalar_t>(), grad_size,
		       target.data_ptr<scalar_t>(), target_size,
		       reduction, log_target);
}

Tensor _kl_div_backward(const Tensor & grad, const Tensor & input,
			const Tensor & target, const int64_t reduction,
			const bool log_target) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_expand = grad.expand_as(input);

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "_kl_div_backward", [&] {
      kl_div_backward_template<scalar_t>(grad_input, grad_expand, target, reduction,
					 log_target);
    });

  return grad_input;
}

#endif // __ARM_FEATURE_SVE

}  // namespace

REGISTER_DISPATCH(kl_div_backward_stub, &_kl_div_backward);

}  // namespace native
}  // namespace at
