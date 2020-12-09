#include <type_traits>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>

#if !defined(__GNUC__) || !defined(__ARM_FEATURE_SVE)

namespace at {
namespace native {

void _bernoulli_ampl_(Tensor& self, double p, c10::optional<Generator> gen) {
  AT_ERROR("_bernoulli_ampl_: ATen not compiled with AMPL support");
}

} // namespace native
} // namespace at

#else

#include <ATen/Parallel.h>

#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>

#include <ATen/native/ampl/ampl.hpp>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/native/Math.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/cpu/DistributionTemplates.h>

namespace at {
namespace native {

void _bernoulli_ampl_(Tensor& self, double p, c10::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  int64_t seed;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(generator->mutex_);
    seed = generator->random();
  }
  int64_t n = self.numel();
  bool contig = self.is_contiguous();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, self.scalar_type(), "_bernoulli_ampl_", [&] {
    at::Tensor tmp_int_tensor;
    if (std::is_same<scalar_t, int>::value && contig) {
      tmp_int_tensor = self;
    } else {
      tmp_int_tensor = at::empty(self.sizes(), self.options().dtype(at::kInt));
    }

    scalar_t *self_ptr = self.data_ptr<scalar_t>();
    int *sample_int_ptr = tmp_int_tensor.data_ptr<int>();

    auto sample = [&](int64_t begin, int64_t end) {
      int64_t len = end - begin;
      if (len > 0) {
	struct StreamStatePtr_ampl stream;
	NewStream_ampl(&stream, 0, seed);
	SkipAheadStream_ampl(stream, begin);
	RngBernoulli_ampl(0, stream, len, sample_int_ptr + begin, p);
	DeleteStream_ampl(stream);

	// vectorized copy if using buffer and contiguous, i.e., being non-int
	// type and contiguous
	if (!std::is_same<scalar_t, int>::value && contig) {
	  scalar_t *self_seg = self_ptr + begin;
	  int* tmp_seg = sample_int_ptr + begin;
	  at::vec256::convert<int, scalar_t>(tmp_seg, self_seg, len);
	}
      }
    };

    parallel_for(0, n, /* grain_size= */ 800, sample);

    // copy_ if using buffer and non contiguous
    if (!contig) {
      self.copy_(tmp_int_tensor);
    }
  });
}

}  // namespace native
}  // namespace at

#endif
