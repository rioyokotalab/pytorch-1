#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/quantized/fake_quant_affine.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <cmath>

/* Fake quantize a tensor
Args:
  output: output tensor.
  input : input tensor.
  sc:  scale to quantize the input tensor to
  zero_point: zero_point
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Fake quantized tensor (float dtype).
*/
namespace at {
namespace native {
#if 1 //Added by Flab (Y. Tamiya)
static inline __host__ __device__
float fake_convert_fp(float x, int ebits, int mbits, int ebias)
{
  if (x == 0.0f || std::isinf(x) || std::isnan(x)) {
    return x;
  } else {
    // Round to Nearest Even Algorithm
    //printf("[DEBUG] ebits=%d,mbits=%d,ebias=%d,x=%f\n",ebits,mbits,ebias,x);
    const int FP32_EBITS = 8;
    const int FP32_MBITS = 23;
#   define BIAS(ebits)     ((1 << ((ebits) -1)) -1)
    uint32_t e_min = BIAS(FP32_EBITS) - BIAS(ebits) + ebias;
    uint32_t e_max = e_min + (1 << ebits) -1;
    //printf("[DEBUG] e_min/max = %d/%d\n", e_min, e_max);
    union {
      uint32_t i;
      float    f;
    } t;
    t.f = x;
    //printf("[DEBUG] x = 0x%08x\n", t.i);

    uint32_t s = (t.i & (1<< (FP32_EBITS+FP32_MBITS))) != 0;
    uint32_t e = (t.i >> FP32_MBITS) & ((1 << FP32_EBITS) -1);
    uint32_t m  = t.i & (((1 << mbits) - 1) << (FP32_MBITS - mbits)); 
    uint32_t r0 = t.i &  (1 << (FP32_MBITS - mbits -1));
    uint32_t r1 = t.i & ((1 << (FP32_MBITS - mbits -1)) -1);

    if (FP32_MBITS - mbits <= 0
	|| r0 == 0
	|| (FP32_MBITS - mbits >= 2 && r1 == 0
	    && (m & (1 << (FP32_MBITS - mbits))) == 0)) {
      ; //floor: nop
    } else {
      // ceil
      if (m == (((1 << mbits) -1) << (FP32_MBITS - mbits))) {
	m = 0;
	e += 1;
      } else {
	m += (1 << (FP32_MBITS - mbits));
      }
    }
    //printf("[DEBUG] s=%d, e=%d, m=%d (before clip)\n", s, e, m);
    if (e > e_max) { e = e_max; m = ((1 << mbits) -1) << (FP32_MBITS - mbits); }
    else if (e < e_min) { e = 0; m = 0; }

    t.i = (s << (FP32_EBITS+FP32_MBITS)) | (e << FP32_MBITS) | m; 
    //printf("[DEBUG] y = 0x%08x\n", t.i);
    return t.f;
  }
}
#endif //Added by Flab (Y. Tamiya)

void fake_quantize_tensor_kernel_cuda(
    Tensor& output,
    const Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // scalar type of this function is guaranteed to be float
//Removed by Flab (Y. Tamiya)//  float inv_scale = 1.0f / scale;
  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.add_output(output);
  iter.add_input(input);
  iter.build();
#if 1 //Added by Flab (Y. Tamiya)
  if (scale == 0.0f) {
  gpu_kernel(iter,
    [zero_point] GPU_LAMBDA (float input_val) -> float {
       return fake_convert_fp(input_val, (zero_point>>8) & 0xff,
			      zero_point & 0xff,
			      (signed char)((zero_point>>16) & 0xff));
    });
  } else {
  float inv_scale = 1.0f / scale;
#endif //Added by Flab (Y. Tamiya)
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float input_val) -> float {
      return (fminf(
                quant_max,
                fmaxf(
                    quant_min,
                    static_cast<int64_t>(std::nearbyint(
                        input_val * inv_scale + zero_point)))) -
            zero_point) *
          scale;
    });
#if 1 //Added by Flab (Y. Tamiya)
  }
#endif //Added by Flab (Y. Tamiya)
}

void fake_quantize_grad_tensor_kernel_cuda(
    Tensor& input_grad,
    const Tensor& input,
    const Tensor& output_grad,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // scalar type of this function is guaranteed to be float
//Removed by Flab (Y. Tamiya)//  float inv_scale = 1.0f / scale;
  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.add_output(input_grad);
  iter.add_input(output_grad);
  iter.add_input(input);
  iter.build();
#if 1 //Added by Flab (Y. Tamiya)
  if (scale == 0.0f) {
  gpu_kernel(iter,
    [zero_point] GPU_LAMBDA (float dy, float x) -> float {
       return fake_convert_fp(dy, (zero_point>>8) & 0xff,
			      zero_point & 0xff,
			      (signed char)((zero_point>>16) & 0xff));
    });
  } else {
  float inv_scale = 1.0f / scale;
#endif //Added by Flab (Y. Tamiya)
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float dy, float x) -> float {
      int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
      return (Xq >= quant_min && Xq <= quant_max) * dy;
    });
#if 1 //Added by Flab (Y. Tamiya)
  }
#endif //Added by Flab (Y. Tamiya)
}

REGISTER_DISPATCH(fake_quant_tensor_stub, &fake_quantize_tensor_kernel_cuda);
REGISTER_DISPATCH(fake_quant_grad_tensor_stub, &fake_quantize_grad_tensor_kernel_cuda);

// Fake quantize per channel

void fake_quant_per_channel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float input_val, float scale, int64_t zero_point) -> float {
#if 1 //Added by Flab (Y. Tamiya)
    if (scale == 0.0f) {
      return fake_convert_fp(input_val, (zero_point>>8) & 0xff,
			     zero_point & 0xff,
			     (signed char)((zero_point>>16) & 0xff));
    } else {
#endif //Added by Flab (Y. Tamiya)
      float inv_scale = 1.0f / scale;
      return (fminf(
                quant_max,
                fmaxf(
                    quant_min,
                    static_cast<int64_t>(std::nearbyint(
                        input_val * inv_scale + zero_point)))) -
            zero_point) *
          scale;
#if 1 //Added by Flab (Y. Tamiya)
    }
#endif //Added by Flab (Y. Tamiya)
    });
}

void fake_quant_grad_per_channel_cuda(TensorIterator &iter, int64_t quant_min, int64_t quant_max) {
  gpu_kernel(iter,
    [=] GPU_LAMBDA (float x, float dy, float scale, int64_t zero_point) -> float {
#if 1 //Added by Flab (Y. Tamiya)
    if (scale == 0.0f) {
      return fake_convert_fp(dy, (zero_point>>8) & 0xff,
			     zero_point & 0xff,
			     (signed char)((zero_point>>16) & 0xff));
    } else {
#endif //Added by Flab (Y. Tamiya)
      float inv_scale = 1.0f / scale;
      int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
      return (Xq >= quant_min && Xq <= quant_max) * dy;
#if 1 //Added by Flab (Y. Tamiya)
    }
#endif //Added by Flab (Y. Tamiya)
    });
}

REGISTER_DISPATCH(fake_quant_per_channel_stub, &fake_quant_per_channel_cuda);
REGISTER_DISPATCH(fake_quant_grad_per_channel_stub, &fake_quant_grad_per_channel_cuda);

} // namespace native
} // namespace at
