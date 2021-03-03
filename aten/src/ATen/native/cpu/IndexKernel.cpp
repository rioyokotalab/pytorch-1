#include <ATen/native/TensorAdvancedIndexing.h>

#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/cpu/AtomicAddFloat.h>

#if defined(__GNUC__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace at { namespace native {
namespace {

using namespace vec256;

struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
          IntArrayRef original_sizes, IntArrayRef original_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides)
    , original_strides(original_strides.data())
    , original_sizes(original_sizes.data()) {
    AT_ASSERT(original_strides.size() == num_indexers);
    AT_ASSERT(original_sizes.size() == num_indexers);
  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;
  const int64_t* original_strides;
  const int64_t* original_sizes;

  int64_t get(int64_t idx) {
    int64_t offset = 0;
    for (int j = 0; j < num_indexers; j++) {
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      int64_t size = original_sizes[j];
      if (value < -size || value >= size) {
        TORCH_CHECK_INDEX(false, "index ", value, " is out of bounds for dimension ", j, " with size ", size);
      }
      if (value < 0) {
        value += size;
      }
      offset += value * original_strides[j];
    }
    return offset;
  }
};

static bool is_constant_index(int ntensor, const int64_t* strides) {
  AT_ASSERT(ntensor >= 3);
  for (int arg = 2; arg < ntensor; arg++) {
    if (strides[arg] != 0) {
      return false;
    }
  }
  return true;
}

template <typename scalar_t, typename func_t>
void cpu_index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride,
                      const func_t& f, bool serial_execution=false)
{
  int ntensor = iter.ntensors();
  // When launch the index parallel version, set a relative samll grain size less than the INTERNAL::GRAIN_SIZE
  // to make the whole available thread numbers get more balanced work load and a better cache location.
  // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
  const int index_parallel_grain_size = 3000;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    char* dst = data[0];
    char* src = data[1];
    if (is_constant_index(ntensor, strides)) {
      // specialization for when every element uses the same index
      int64_t offset = indexer.get(0);
      if (strides[0] == sizeof(scalar_t) && strides[1] == sizeof(scalar_t)) {
        for (int64_t i = 0; i < n; i++) {
          f(dst + strides[0] * i, src + strides[1] * i, offset);
        }
      } else {
        for (int64_t i = 0; i < n; i++) {
          f(dst + strides[0] * i, src + strides[1] * i, offset);
        }
      }
    } else {
      for (int64_t i = 0; i < n; i++) {
        int64_t offset = indexer.get(i);
        f(dst + strides[0] * i, src + strides[1] * i, offset);
      }
    }
  };
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    iter.for_each(loop, index_parallel_grain_size);
  }
}

void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "index_cpu", [&] {
    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
      *(scalar_t*)dst = *(scalar_t*)(src + offset);
    });
  });
}

void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16,
    iter.dtype(), "index_put", [&] {
    if (accumulate) {
      bool use_parallel_for = ((iter.numel() >= internal::GRAIN_SIZE) && (at::get_num_threads() > 1));
      if (iter.dtype() == ScalarType::Float && use_parallel_for) {
        cpu_index_kernel<float>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
          cpu_atomic_add_float((float*)(dst + offset), *(float*)src);
        });
      } else {
        // TODO: investigate parallelization of the accumulate kernel. Unlike the non-accumulate case,
        // this needs to be thread-safe.
        cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset) += *(scalar_t*)src;
        }, /*serial_execution=*/true);
      }
    } else {
      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
        *(scalar_t*)(dst + offset) = *(scalar_t*)src;
      });
    }
  });
}

template <typename scalar_t, typename mask_t>
void cpu_masked_fill_kernel(TensorIterator& iter, scalar_t value) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* mask = data[1];
    for (int64_t i = 0; i < n; i++) {
      mask_t mask_value = *(mask_t*)(mask + strides[1] * i);
      if (!is_mask_bool) {
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
      }
      if (mask_value) {
        *(scalar_t*)(dst + strides[0] * i) = value;
      }
    }
  };
  iter.for_each(loop);
}

void masked_fill_kernel(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
    iter.dtype(), "masked_fill", [&] {
      scalar_t scalar_val = value.to<scalar_t>();
      auto mask_dtype = iter.input_dtype(0);
      if (mask_dtype == ScalarType::Bool) {
        cpu_masked_fill_kernel<scalar_t, bool>(iter, scalar_val);
      } else {
        cpu_masked_fill_kernel<scalar_t, unsigned char>(iter, scalar_val);
      }
    });
}

template <typename scalar_t, typename mask_t, typename func_t>
void cpu_masked_select_serial_kernel(TensorIterator& iter, const func_t& f) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  int64_t offset = 0;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* src = data[1];
    char* mask = data[2];
    for (int64_t i = 0; i < n; i++) {
      mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
      if (!is_mask_bool) {
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
      }
      if (mask_value) {
        int64_t offset_bytes = offset * sizeof(scalar_t);
        f(dst, src + strides[1] * i, offset_bytes);
        offset++;
      }
    }
  };
  iter.serial_for_each(loop, {0, iter.numel()});
}

void masked_select_serial_kernel(TensorIterator& iter, int64_t result_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
    iter.dtype(), "masked_select", [&] {
      auto mask_dtype = iter.input_dtype(1);
      if (mask_dtype == ScalarType::Bool) {
        cpu_masked_select_serial_kernel<scalar_t, bool>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      } else {
        cpu_masked_select_serial_kernel<scalar_t, unsigned char>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      }
    });
}

template <typename scalar_t, typename mask_t, typename func_t>
void cpu_masked_select_kernel(TensorIterator& iter, const func_t& f) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* src = data[1];
    char* mask = data[2];
    char* mask_prefix_sum = data[3];
    for (int64_t i = 0; i < n; i++) {
      mask_t mask_value = *(mask_t*)(mask + strides[2] * i);
      if (!is_mask_bool) {
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
      }
      if (mask_value) {
        int64_t offset = *(int64_t*)(mask_prefix_sum + strides[3] * i);
        int64_t offset_bytes = (offset - 1) * sizeof(scalar_t);
        f(dst, src + strides[1] * i, offset_bytes);
      }
    }
  };
  iter.for_each(loop);
}

void masked_select_kernel(TensorIterator& iter, int64_t result_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Bool, ScalarType::BFloat16, ScalarType::Half,
    iter.dtype(), "masked_select", [&] {
      auto mask_dtype = iter.input_dtype(1);
      if (mask_dtype == ScalarType::Bool) {
        cpu_masked_select_kernel<scalar_t, bool>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      } else {
        cpu_masked_select_kernel<scalar_t, unsigned char>(iter, [result_stride](char* dst, char* src, int64_t offset) {
          *(scalar_t*)(dst + offset*result_stride) = *(scalar_t*)src;
        });
      }
    });
}

#if defined(__GNUC__) && defined(__ARM_FEATURE_SVE)
#define NONZERO_OUT_1D_S32_SVE_TEMPLATE(stype, vtype, bit_stype, zero_value)                \
template <typename scalar_t,						                    \
	  typename std::enable_if<std::is_same<scalar_t, stype>::value, int>::type = 0>     \
void nonzero_out_1d_impl(Tensor & result, const Tensor & self) {                            \
  TORCH_CHECK(result.dtype() == kLong, "output indices must be of scalar type Long");       \
  const int64_t numel = self.numel();                                                       \
  Tensor result_buf =                                                                       \
    at::empty({numel, 1}, self.options().dtype(at::kInt));                                  \
  int64_t total_index = 0;                                                                  \
  vtype zero = svdup_n_##bit_stype##32(zero_value);                                         \
  svint32_t zero_s32 = svdup_n_s32(0);                                                      \
  const scalar_t* self_ptr = self.data_ptr<scalar_t>();                                     \
  int32_t* result_buf_ptr = result_buf.data_ptr<int32_t>();                                 \
  const int64_t fraction = numel % svcntw();                                                \
  svbool_t pg_true = svptrue_b32();                                                         \
  for (int64_t i = 0; i < numel - fraction; i += svcntw()) {		                    \
    svbool_t mask =                                                                         \
      svcmpne_##bit_stype##32(pg_true, svld1_##bit_stype##32(pg_true, self_ptr + i), zero); \
    svint32_t index_base = svindex_s32(i, 1);                                               \
    svint32_t index = svcompact_s32(mask, index_base);                                      \
    int64_t nonzero = (int64_t)svcntp_b32(pg_true, mask);                                   \
    svbool_t pg = svwhilelt_b32(0ull, nonzero);                                             \
    svst1_s32(pg, result_buf_ptr + total_index, index);                                     \
    total_index += nonzero;                                                                 \
  }                                                                                         \
  for (int64_t i = numel - fraction; i < numel; i += svcntw()) {                            \
    svbool_t pg = svwhilelt_b32(i, numel);                                                  \
    svbool_t mask =                                                                         \
      svcmpne_##bit_stype##32(pg, svld1_##bit_stype##32(pg, self_ptr + i), zero);           \
    svint32_t index_base = svindex_s32(i, 1);                                               \
    svint32_t index = svcompact_s32(mask, index_base);                                      \
    int64_t nonzero = (int64_t)svcntp_b32(pg, mask);                                        \
    pg = svwhilelt_b32(0ull, nonzero);                                                      \
    svst1_s32(pg, result_buf_ptr + total_index, index);                                     \
    total_index += nonzero;                                                                 \
  }                                                                                         \
  result_buf.resize_({total_index, 1});                                                     \
  result.resize_({total_index, 1});                                                         \
  result.copy_(result_buf);                                                                 \
}

NONZERO_OUT_1D_S32_SVE_TEMPLATE(float, svfloat32_t, f, 0.f)
NONZERO_OUT_1D_S32_SVE_TEMPLATE(int32_t, svint32_t, s, 0)
#else
template <typename scalar_t>
void nonzero_out_1d_impl(Tensor &, const Tensor &) {
}
#endif // defined(__GNUC__) && defined(__ARM_FEATURE_SVE)

Tensor& _nonzero_(Tensor & result, const Tensor & self) {
  if (self.numel() <= INT32_MAX) {
    if (self.is_contiguous() && self.dim() == 1) {
      if (self.scalar_type() != kLong && isIntegralType(self.scalar_type(), true)) {
        nonzero_out_1d_impl<int32_t>(result, self.to(at::kInt));
        return result;
      } else if(self.scalar_type() == kFloat) {
        nonzero_out_1d_impl<float>(result, self);
        return result;
      }
    }
  }
  return at::native::legacy::cpu::_th_nonzero_out(result, self);
}

} // anonymous namespace

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);
REGISTER_DISPATCH(masked_fill_stub, &masked_fill_kernel);
REGISTER_DISPATCH(masked_select_serial_stub, &masked_select_serial_kernel);
REGISTER_DISPATCH(masked_select_stub, &masked_select_kernel);
REGISTER_DISPATCH(nonzero_stub, &_nonzero_);

}} // namespace at::native
