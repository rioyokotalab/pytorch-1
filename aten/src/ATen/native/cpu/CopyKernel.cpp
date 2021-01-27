#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/TypeCast.h>

namespace at {
namespace native {
namespace {

static void copy_kernel(TensorIterator& iter, bool non_blocking) {
  ScalarType dtype = iter.dtype(0);
  if (dtype == iter.dtype(1)) {
    if (dtype == ScalarType::Half) {
      cpu_kernel(iter, [=](at::Half a) -> at::Half { return a; });
    } else if (dtype == ScalarType::BFloat16) {
      cpu_kernel(iter, [=](at::BFloat16 a) -> at::BFloat16 { return a; });
    } else if (isQIntType(dtype)) {
      AT_DISPATCH_QINT_TYPES(dtype, "copy_kernel", [&] {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return a; },
            [=](Vec256<scalar_t> a) -> Vec256<scalar_t> { return a; });
      });
    } else if (isComplexType(dtype)) {
      AT_DISPATCH_COMPLEX_TYPES(dtype, "copy_kernel", [&] {
          cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return a; },
            [=](Vec256<scalar_t> a) -> Vec256<scalar_t> { return a; });
        });
    } else {
      AT_DISPATCH_ALL_TYPES_AND(
          ScalarType::Bool, dtype, "copy_kernel", [&] {
            cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return a; },
                [=](Vec256<scalar_t> a) { return a; });
          });
    }
#if defined(__GNUC__) && defined(__ARM_FEATURE_SVE)
  } else if (iter.is_contiguous() &&
	     ((dtype == ScalarType::Half && iter.dtype(1) == ScalarType::Float) ||
	      (dtype == ScalarType::Float && iter.dtype(1) == ScalarType::Half))) {
    if (dtype == ScalarType::Half && iter.dtype(1) == ScalarType::Float) {
      at::Half *dst_ptr = iter.tensor(0).data_ptr<at::Half>();
      const float *src_ptr = iter.tensor(1).data_ptr<float>();
      at::parallel_for(0, iter.numel(), 2048, [&](int64_t start, int64_t end) {
	at::vec256::convert<float, at::Half>(src_ptr + start, dst_ptr + start, end - start);
      });
    } else if (dtype == ScalarType::Float && iter.dtype(1) == ScalarType::Half) {
      float *dst_ptr = iter.tensor(0).data_ptr<float>();
      const at::Half *src_ptr = iter.tensor(1).data_ptr<at::Half>();
      at::parallel_for(0, iter.numel(), 2048, [&](int64_t start, int64_t end) {
	at::vec256::convert<at::Half, float>(src_ptr + start, dst_ptr + start, end - start);
      });
    }
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, dtype, "copy_", [&] {
      using dest_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, iter.dtype(1), "copy_", [&] {
        // Note (@zasdfgbnm):
        //
        // The code below can not be simplified as
        //    cpu_kernel(iter, c10::static_cast_with_inter_type<dest_t, scalar_t>::apply);
        //
        // because this would force the compiler to instantiate the inline function and generate a function call in the loop
        // instead of inlining it, making all the optimizations like vectorization impossible.
        // You can verify this by looking the the symbols of `libtorch_cpu.so`:
        //
        //    readelf -Ws libtorch_cpu.so | grep static_cast_with_inter_type
        //
        // If done correctly, the above command should have no output.
        //
        // See: https://github.com/pytorch/pytorch/issues/31271
        cpu_kernel(iter, [](scalar_t src) -> dest_t {
          return c10::static_cast_with_inter_type<dest_t, scalar_t>::apply(src); });
      });
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(copy_stub, &copy_kernel);

} // namespace native
} // namespace at
