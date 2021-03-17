#ifdef __ARM_FEATURE_SVE

#ifndef AMPL_ACTIVATION_HPP_
#define AMPL_ACTIVATION_HPP_

#include <cmath>
#include <ATen/native/ampl/ampl.h>

namespace ampl {

/**
 * Computes the GELU forward value of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] x Pointer to an array that contains the input vector x.
 * @param[out] y Pointer to an array that contains the output vector y.
 */
template <typename scalar_t>
void GeluForward(
    const int64_t n,
    const scalar_t* x,
    scalar_t* y);

/**
 * Computes the GELU backward value of vector elements.
 *
 * @param[in] n Specifies the number of elements to be calculated.
 * @param[in] x Pointer to an array that contains the input vector x.
 * @param[in] dy Pointer to an array that contains the input vector dy.
 * @param[out] dx Pointer to an array that contains the output vector dx.
 */
template <typename scalar_t>
void GeluBackward(
    const int64_t n,
    const scalar_t* x,
    const scalar_t* dy,
    scalar_t* dx);

} // namespace ampl

#endif // AMPL_ACTIVATION_HPP_
#endif // __ARM_FEATURE_SVE
