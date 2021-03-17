#ifdef __ARM_FEATURE_SVE

#ifndef AMPL_UNARY_OPS_HPP_
#define AMPL_UNARY_OPS_HPP_

#include <ATen/native/ampl/ampl.h>

namespace ampl {

/**
 * Computes an exponential of SVE data type elements.
 *
 * @param[in] pg Predicate for bits control.
 * @param[in] a Input SVE data type vector.
 * @return Exp(a)
 * @attention Support types are svfloat16_t, svfloat32_t, svfloat64_t
 */
template <typename svbase_t>
inline svbase_t Exp(const svbool_t& pg, const svbase_t& a);

/**
 * Computes an exponential of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] a Pointer to an array that contains the input vector a.
 * @param[out] y Pointer to an array that contains the output vector y.
 * @attention Support types are float16_t, float32_t, float64_t
 */
template <typename scalar_t>
void vExp(const int64_t n, const scalar_t* a, scalar_t* y);

/**
 * Computes an error function of SVE data type elements.
 *
 * @param[in] pg Predicate for bits control.
 * @param[in] a Input SVE data type vector.
 * @return Erf(a)
 * @attention Support types are svfloat16_t, svfloat32_t, svfloat64_t
 */
template <typename svbase_t>
inline svbase_t Erf(const svbool_t& pg, const svbase_t& a);

/**
 * Computes the error function value of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] a Pointer to an array that contains the input vector a.
 * @param[out] y Pointer to an array that contains the output vector y.
 * @attention Support types are float16_t, float32_t, float64_t
 */
template <typename scalar_t>
void vErf(const int64_t n, const scalar_t* a, scalar_t* y);

} // namespace ampl

#endif // AMPL_UNARY_OPS_HPP_
#endif // __ARM_FEATURE_SVE
