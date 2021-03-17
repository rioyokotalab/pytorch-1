#ifdef __ARM_FEATURE_SVE

#include <ATen/native/ampl/common.hpp>
#include <ATen/native/ampl/unary_ops.hpp>
#include <ATen/native/ampl/unary_ops_impl.hpp>

namespace ampl {

// Implement unary ops with N-bit
#define AMPL_IMPLEMENT_VEC_UOPS(func, bit)                           \
  template <>                                                        \
  void v##func<float##bit##_t>(                                      \
      const int64_t n, const float##bit##_t* a, float##bit##_t* y) { \
    parallel_for(                                                    \
        0,                                                           \
        n,                                                           \
        2048,                                                        \
        GetVectorNumel<float##bit##_t>(),                            \
        [&](int64_t begin, int64_t end) {                            \
          for (int64_t i = begin; i < end;                           \
               i += GetVectorNumel<float##bit##_t>()) {              \
            const svbool_t pg = svwhilelt_b##bit(i, end);            \
            const svfloat##bit##_t src = svld1_f##bit(pg, a + i);    \
            svfloat##bit##_t dst = func(pg, src);                    \
            svst1_f##bit(pg, y + i, dst);                            \
          }                                                          \
        });                                                          \
  }

AMPL_IMPLEMENT_VEC_UOPS(Exp, 16)
AMPL_IMPLEMENT_VEC_UOPS(Exp, 32)
AMPL_IMPLEMENT_VEC_UOPS(Exp, 64)
AMPL_IMPLEMENT_VEC_UOPS(Erf, 16)
AMPL_IMPLEMENT_VEC_UOPS(Erf, 32)
AMPL_IMPLEMENT_VEC_UOPS(Erf, 64)

#undef AMPL_IMPLEMENT_VEC_UOPS

} // namespace ampl

// Define C API wrapping C++ API
#define AMPL_DEFINE_C_API(op, type, ampldtype)                            \
  sv##type ampldtype##op##_ampl(const svbool_t pg, const sv##type a) {    \
    return ampl::op(pg, a);                                               \
  }                                                                       \
  void v##ampldtype##op##_ampl(const int64_t n, const type* a, type* y) { \
    ampl::v##op(n, a, y);                                                 \
  }

AMPL_DEFINE_C_API(Exp, float16_t, h)
AMPL_DEFINE_C_API(Exp, float32_t, s)
AMPL_DEFINE_C_API(Exp, float64_t, d)
AMPL_DEFINE_C_API(Erf, float16_t, h)
AMPL_DEFINE_C_API(Erf, float32_t, s)
AMPL_DEFINE_C_API(Erf, float64_t, d)

#undef AMPL_DEFINE_C_API

#endif // __ARM_FEATURE_SVE
