#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

/*
  KlDiv Backward operator
*/

namespace at {
namespace native {

using kl_div_backward_fn =
  Tensor (*)(const Tensor &, const Tensor &, const Tensor &, int64_t, bool);

DECLARE_DISPATCH(kl_div_backward_fn, kl_div_backward_stub);

}  // namespace native
}  // namespace at
