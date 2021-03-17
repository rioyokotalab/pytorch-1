#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

/*
  baddbmm blas operator
*/

namespace at {
namespace native {

using baddbmm_blas_fn =
  Tensor& (*)(Tensor&, const Tensor&, const Tensor&, Scalar, Scalar);

DECLARE_DISPATCH(baddbmm_blas_fn, baddbmm_blas_stub);

}  // namespace native
}  // namespace at
