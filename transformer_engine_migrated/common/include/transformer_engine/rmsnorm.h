/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file rmsnorm.h
 *  \brief RMSNorm functions.
 */

#ifndef TRANSFORMER_ENGINE_RMSNORM_H_
#define TRANSFORMER_ENGINE_RMSNORM_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute RMSNorm on the input.
 *
 * The formula used:
 * @f[
 * y = \frac{x}{RMS_\varepsilon(x)}\gamma
 * @f]
 * where
 * @f[
 * RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} x_i^2 + \varepsilon}
 * @f]
 *
 * Calling this function with workspace and barrier set to empty tensor will not
 * perform the operation, but instead set the shape and type of the workspace
 * and barrier tensors to the required values.
 *
 *  \param[in]     x                   Input tensor of shape [N, H].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[in]     epsilon             Value added to denominator for numerical stability.
 *  \param[in,out] z                   Output tensor of shape [N, H].
 *  \param[out]    rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[in]     stream              CUDA stream used for the operation.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[out]    barrier             Barrier tensor.
 */
void nvte_rmsnorm_fwd(const NVTETensor x, const NVTETensor gamma,
                      const float epsilon, NVTETensor z, NVTETensor rsigma,
                      dpct::queue_ptr stream, const int multiprocessorCount,
                      NVTETensor workspace, NVTETensor barrier);

/*! \brief Compute RMSNorm with zero-centered gamma on the input.
 *
 * The formula used:
 * @f[
 * y = \frac{x}{RMS_\varepsilon(x)}(1 + \gamma)
 * @f]
 * where
 * @f[
 * RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} x_i^2 + \varepsilon}
 * @f]
 *
 * Calling this function with workspace and barrier set to empty tensor will not
 * perform the operation, but instead set the shape and type of the workspace
 * and barrier tensors to the required values.
 *
 *  \param[in]     x                   Input tensor of shape [N, H].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[in]     epsilon             Value added to denominator for numerical stability.
 *  \param[in,out] z                   Output tensor of shape [N, H].
 *  \param[out]    rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[in]     stream              CUDA stream used for the operation.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[out]    barrier             Barrier tensor.
 */
void nvte_rmsnorm1p_fwd(const NVTETensor x, const NVTETensor gamma,
                        const float epsilon, NVTETensor z, NVTETensor rsigma,
                        dpct::queue_ptr stream, const int multiprocessorCount,
                        NVTETensor workspace, NVTETensor barrier);

/*! \brief Compute backward of RMSNorm.
 *
 * This function computes the gradient of function:
 * @f[
 * y = \frac{x}{RMS_\varepsilon(x)}\gamma
 * @f]
 * where
 * @f[
 * RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} x_i^2 + \varepsilon}
 * @f]
 * with respect to \f$x\f$ and \f$gamma\f$.
 *
 * Calling this function with workspace, barrier, dgamma_part set
 * to empty tensor will not perform the operation, but instead set the shape and type
 * of these tensors to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [N, H].
 *  \param[in]     x                   Forward input tensor of shape [N, H].
 *  \param[in]     rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[out]    dx                  Output gradient of shape [N, H].
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H].
 *  \param[out]    dgamma_part         Storage for partial gamma gradient.
 *  \param[in]     stream              CUDA stream used for the operation.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[out]    barrier             Barrier tensor.
 */
void nvte_rmsnorm_bwd(const NVTETensor dz, const NVTETensor x,
                      const NVTETensor rsigma, const NVTETensor gamma,
                      NVTETensor dx, NVTETensor dgamma, NVTETensor dgamma_part,
                      dpct::queue_ptr stream, const int multiprocessorCount,
                      NVTETensor workspace, NVTETensor barrier);

/*! \brief Compute backward of RMSNorm with zero-centered gamma.
 *
 * This function computes the gradient of function:
 * @f[
 * y = \frac{x}{RMS_\varepsilon(x)}(1 + \gamma)
 * @f]
 * where
 * @f[
 * RMS_\varepsilon(x) = \sqrt{\frac{1}{n}\sum_{i=0}^{n-1} x_i^2 + \varepsilon}
 * @f]
 * with respect to \f$x\f$ and \f$gamma\f$.
 *
 * Calling this function with workspace, barrier, dgamma_part set
 * to empty tensor will not perform the operation, but instead set the shape and type
 * of these tensors to the required values.
 *
 *  \param[in]     dz                  Incoming gradient tensor of shape [N, H].
 *  \param[in]     x                   Forward input tensor of shape [N, H].
 *  \param[in]     rsigma              Reciprocal of the root mean square of the input
 *                                     calculated over the last dimension. Shape: [N].
 *  \param[in]     gamma               Gamma tensor of shape [H].
 *  \param[out]    dx                  Output gradient of shape [N, H].
 *  \param[out]    dgamma              Gradient for gamma tensor of shape [H].
 *  \param[out]    dgamma_part         Storage for partial gamma gradient.
 *  \param[in]     stream              CUDA stream used for the operation.
 *  \param[in]     multiprocessorCount Number of SMs in the device.
 *  \param[out]    workspace           Workspace tensor.
 *  \param[out]    barrier             Barrier tensor.
 */
void nvte_rmsnorm1p_bwd(const NVTETensor dz, const NVTETensor x,
                        const NVTETensor rsigma, const NVTETensor gamma,
                        NVTETensor dx, NVTETensor dgamma,
                        NVTETensor dgamma_part, dpct::queue_ptr stream,
                        const int multiprocessorCount, NVTETensor workspace,
                        NVTETensor barrier);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_RMSNORM_H_
