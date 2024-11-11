/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include "cudnn_frontend_ConvDesc.h"
#include "cudnn_frontend_PointWiseDesc.h"
#include "cudnn_frontend_MatMulDesc.h"
#include "cudnn_frontend_ReductionDesc.h"
#include "cudnn_frontend_Resample.h"
#include "cudnn_frontend_Rng.h"
#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// Operation_v8 Class
/// This class has the properties of the operation
/// Properties:
///    - xDesc
///    - yDesc
///    - wdesc
///    - bdesc
///    - tDesc
///    - dydesc
///    - dxdesc
///    - cdesc
///    - amatdesc
///    - bmatdesc
///    - cmatdesc
///    - moverridedesc
///    - noverridedesc
///    - koverridedesc
///    - pwdesc
///    - matmuldesc
///    - reductiondesc
///    - flagdesc
///    - inputDescs
///    - alpha
///    - beta
///    - alpha2
///    - axis
///    - inplaceIndex
///    - mode
///    - value
///
/// Use OperationBuilder_v8 to build this class.
/// Describe returns a string describing the convolution operation
///
class Operation_v8 : public BackendDescriptor {
   public:
    friend class OperationBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_OPERATION :"
           << " OpMode: " << op_mode;
        ss << std::hex << " X " << xdesc;
        ss << std::hex << " Y " << ydesc;
        ss << std::hex << " W " << wdesc;
        ss << std::hex << " B " << bdesc;
        ss << std::hex << " T " << tdesc;
        ss << std::hex << " DW " << dwdesc;
        ss << std::hex << " DY " << dydesc;
        ss << std::hex << " DX " << dxdesc;
        ss << std::hex << " C " << cdesc;
        ss << std::hex << " A Mtrix " << amatdesc;
        ss << std::hex << " B Mtrix " << bmatdesc;
        ss << std::hex << " C Mtrix " << cmatdesc;
        ss << std::hex << " P " << pwdesc;
        ss << std::hex << " MatMul " << matmuldesc;
        ss << std::hex << " Reduction " << reductiondesc;
        ss << std::dec << " alphabetaType " << alphabetaType;
        ss << " Alpha: " << alpha_s << " " << alpha_d;
        ss << " Alpha2: " << alpha2_s << " " << alpha2_d;
        ss << " Beta: " << beta_s << " " << beta_d;
        return ss.str();
    }

    Operation_v8(Operation_v8 &&from) = default;
    Operation_v8 &
    operator=(Operation_v8 &&from) = default;

    // Will be deprecated. Do Not use
    ManagedOpaqueDescriptor
    getOutputTensor() {
        return (op_mode == DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR) ? cmatdesc : ydesc;
    }

    std::string const &
    getTag() const {
        return operationTag;
    }

    feature_vector_t
    getFeatureVector() const {
        return feature_vector;
    }

    ~Operation_v8() = default;

   private:
    Operation_v8()                     = default;
    Operation_v8(Operation_v8 const &) = delete;
    Operation_v8 &
    operator=(Operation_v8 const &) = delete;

    DescriptorType_t op_mode = DescriptorType_t::NOT_SET;

    ManagedOpaqueDescriptor xdesc              = nullptr;
    ManagedOpaqueDescriptor ydesc              = nullptr;
    ManagedOpaqueDescriptor wdesc              = nullptr;
    ManagedOpaqueDescriptor bdesc              = nullptr;
    ManagedOpaqueDescriptor tdesc              = nullptr;
    ManagedOpaqueDescriptor dydesc             = nullptr;
    ManagedOpaqueDescriptor dxdesc             = nullptr;
    ManagedOpaqueDescriptor dwdesc             = nullptr;
    ManagedOpaqueDescriptor cdesc              = nullptr;
    ManagedOpaqueDescriptor resampledesc       = nullptr;
    ManagedOpaqueDescriptor rngdesc            = nullptr;
    ManagedOpaqueDescriptor amatdesc           = nullptr;
    ManagedOpaqueDescriptor bmatdesc           = nullptr;
    ManagedOpaqueDescriptor cmatdesc           = nullptr;
    ManagedOpaqueDescriptor moverridedesc      = nullptr;
    ManagedOpaqueDescriptor noverridedesc      = nullptr;
    ManagedOpaqueDescriptor koverridedesc      = nullptr;
    ManagedOpaqueDescriptor pwdesc             = nullptr;
    ManagedOpaqueDescriptor matmuldesc         = nullptr;
    ManagedOpaqueDescriptor reductiondesc      = nullptr;
    ManagedOpaqueDescriptor sumdesc            = nullptr;
    ManagedOpaqueDescriptor sqsumdesc          = nullptr;
    ManagedOpaqueDescriptor scaledesc          = nullptr;
    ManagedOpaqueDescriptor biasdesc           = nullptr;
    ManagedOpaqueDescriptor dscaledesc         = nullptr;
    ManagedOpaqueDescriptor dbiasdesc          = nullptr;
    ManagedOpaqueDescriptor eqscaledesc        = nullptr;
    ManagedOpaqueDescriptor eqscaledesc1       = nullptr;
    ManagedOpaqueDescriptor eqbiasdesc         = nullptr;
    ManagedOpaqueDescriptor prevMeandesc       = nullptr;
    ManagedOpaqueDescriptor prevVardesc        = nullptr;
    ManagedOpaqueDescriptor nextMeandesc       = nullptr;
    ManagedOpaqueDescriptor nextVardesc        = nullptr;
    ManagedOpaqueDescriptor savedMeandesc      = nullptr;
    ManagedOpaqueDescriptor savedInVardesc     = nullptr;
    ManagedOpaqueDescriptor accumCountdesc     = nullptr;
    ManagedOpaqueDescriptor epsilondesc        = nullptr;
    ManagedOpaqueDescriptor expDecayFactordesc = nullptr;
    ManagedOpaqueDescriptor idxdesc            = nullptr;
    ManagedOpaqueDescriptor offsetdesc         = nullptr;
    ManagedOpaqueDescriptor seeddesc           = nullptr;
    std::vector<ManagedOpaqueDescriptor> peerStatdescs;

    /*
    DPCT1007:367: Migration of CUDNN_TYPE_FLOAT is not supported.
    */
    cudnnBackendAttributeType_t alphabetaType = CUDNN_TYPE_FLOAT;
    dpct::library_data_t compute_type         = dpct::library_data_t::real_float;
    /*
    DPCT1007:368: Migration of CUDNN_GENSTATS_SUM_SQSUM is not supported.
    */
    cudnnGenStatsMode_t genstats_mode = CUDNN_GENSTATS_SUM_SQSUM;
    /*
    DPCT1007:369: Migration of CUDNN_BN_FINALIZE_STATISTICS_TRAINING is not supported.
    */
    cudnnBnFinalizeStatsMode_t bn_stats_mode = CUDNN_BN_FINALIZE_STATISTICS_TRAINING;

    NormFwdPhase_t norm_fwd_phase;
    NormMode_t norm_mode;

    float alpha_s = 1.0f, beta_s = .0f, alpha2_s = 1.0f;
    double alpha_d = 1.0, beta_d = 0.0, alpha2_d = 1.0;
    int64_t pointwise_port_count        = -1;
    PointwiseMode_t pointwise_mode      = PointwiseMode_t::NOT_SET;
    bool is_pointwise_activation_fwd_op = false;
    bool is_pointwise_identity_op       = false;
    bool is_pointwise_activation_bwd_op = false;
    bool is_pointwise_math_op           = false;
    std::string operationTag;
    feature_vector_t feature_vector;
    int64_t seed = 0;
};

///
/// OperationBuilder_v8 Class
/// Helper class used to build Operation_v8 class

class OperationBuilder_v8 {
   private:
    Operation_v8 m_operation;
    bool is_convolution_op   = false;
    bool is_pointwise_op     = false;
    bool is_matmul_op        = false;
    bool is_reduction_op     = false;
    bool is_genstats_op      = false;
    bool is_bn_finalize_op   = false;
    bool is_resample_fwd_op  = false;
    bool is_resample_bwd_op  = false;
    bool is_norm_forward_op  = false;
    bool is_norm_backward_op = false;
    bool is_bn_bwd_weight    = false;
    bool is_rng_op           = false;
    bool is_reshape_op       = false;

    using Message_t = const char *;

    int64_t xTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t xTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t wTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t wTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t yTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t yTensor_strA[CUDNN_DIM_MAX + 1];
    int64_t idxTensor_dimA[CUDNN_DIM_MAX + 1];
    int64_t idxTensor_strA[CUDNN_DIM_MAX + 1];

    bool is2D = true;

    int64_t conv_padding[CUDNN_DIM_MAX];
    int64_t conv_dilation[CUDNN_DIM_MAX];
    int64_t conv_stride[CUDNN_DIM_MAX];
    int64_t mode;
    int64_t xType, yType, wType, cType, idxType /* compute_precision */;

    int64_t tensor_dims = 0;

    Operation_v8 &&
    build_reduction_op() {
        m_operation.operationTag = "Reduction";
        auto status              = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:370: Migration of CUDNN_ATTR_OPERATION_REDUCTION_DESC is not supported.
            */
            CUDNN_ATTR_OPERATION_REDUCTION_DESC,
            /*
            DPCT1007:371: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.reductiondesc->get_backend_descriptor()));

        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_DESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:372: Migration of CUDNN_ATTR_OPERATION_REDUCTION_XDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_REDUCTION_XDESC,
            /*
            DPCT1007:373: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.xdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_XDESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:374: Migration of CUDNN_ATTR_OPERATION_REDUCTION_YDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_REDUCTION_YDESC,
            /*
            DPCT1007:375: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.ydesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_REDUCTION_YDESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_matmul_op() {
        m_operation.operationTag = "Matmul";
        auto status              = 0;
        status =
            cudnn_frontend::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:376: Migration of CUDNN_ATTR_OPERATION_MATMUL_ADESC is not supported.
                                          */
                                          CUDNN_ATTR_OPERATION_MATMUL_ADESC,
                                          /*
                                          DPCT1007:377: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &(m_operation.amatdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_ADESC Failed");
            return std::move(m_operation);
        }
        status =
            cudnn_frontend::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:378: Migration of CUDNN_ATTR_OPERATION_MATMUL_BDESC is not supported.
                                          */
                                          CUDNN_ATTR_OPERATION_MATMUL_BDESC,
                                          /*
                                          DPCT1007:379: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &(m_operation.bmatdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_BDESC Failed");
            return std::move(m_operation);
        }
        status =
            cudnn_frontend::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:380: Migration of CUDNN_ATTR_OPERATION_MATMUL_CDESC is not supported.
                                          */
                                          CUDNN_ATTR_OPERATION_MATMUL_CDESC,
                                          /*
                                          DPCT1007:381: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &(m_operation.cmatdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_CDESC Failed");
            return std::move(m_operation);
        }
#if (CUDNN_VERSION >= 8700)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8700, m_operation, "CUDNN_BACKEND_OPERATION: M,N,K override Requires cudnn 8.7.0 and above");
        if (m_operation.moverridedesc != nullptr) {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:382: Migration of CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC,
                /*
                DPCT1007:383: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.moverridedesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC Failed");
                return std::move(m_operation);
            }
        }
        if (m_operation.noverridedesc != nullptr) {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:384: Migration of CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC,
                /*
                DPCT1007:385: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.noverridedesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC Failed");
                return std::move(m_operation);
            }
        }
        if (m_operation.koverridedesc != nullptr) {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:386: Migration of CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC,
                /*
                DPCT1007:387: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.koverridedesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC Failed");
                return std::move(m_operation);
            }
        }
#endif
        status =
            cudnn_frontend::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:388: Migration of CUDNN_ATTR_OPERATION_MATMUL_DESC is not supported.
                                          */
                                          CUDNN_ATTR_OPERATION_MATMUL_DESC,
                                          /*
                                          DPCT1007:389: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &(m_operation.matmuldesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_MATMUL_DESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_pointwise_op() {
        auto status = 0;

        json j                   = m_operation.pointwise_mode;
        m_operation.operationTag = j;

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:390: Migration of CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR is not supported.
            */
            CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
            /*
            DPCT1007:391: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.pwdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:392: Migration of CUDNN_ATTR_OPERATION_POINTWISE_XDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
            /*
            DPCT1007:393: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.xdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_XDESC Failed");
            return std::move(m_operation);
        }

        if (!m_operation.is_pointwise_activation_bwd_op) {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:394: Migration of CUDNN_ATTR_OPERATION_POINTWISE_YDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                /*
                DPCT1007:395: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.ydesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_YDESC Failed");
                return std::move(m_operation);
            }
        } else {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:396: Migration of CUDNN_ATTR_OPERATION_POINTWISE_DYDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_POINTWISE_DYDESC,
                /*
                DPCT1007:397: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.dydesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_DYDESC Failed");
                return std::move(m_operation);
            }

            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:398: Migration of CUDNN_ATTR_OPERATION_POINTWISE_DXDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_POINTWISE_DXDESC,
                /*
                DPCT1007:399: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.dxdesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_DXDESC Failed");
                return std::move(m_operation);
            }
        }

        /*
        DPCT1007:400: Migration of CUDNN_TYPE_FLOAT is not supported.
        */
        void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                     : static_cast<void *>(&m_operation.alpha_d));
        /*
        DPCT1007:401: Migration of CUDNN_TYPE_FLOAT is not supported.
        */
        void *alpha2 = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha2_s)
                                                                      : static_cast<void *>(&m_operation.alpha2_d));
        status       = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:402: Migration of CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 is not supported.
            */
            CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
            m_operation.alphabetaType,
            1,
            alpha);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:403: Migration of CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 is not supported.
            */
            CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
            m_operation.alphabetaType,
            1,
            alpha2);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 Failed");
            return std::move(m_operation);
        }

        if (m_operation.pointwise_port_count >= 3 && !m_operation.is_pointwise_activation_bwd_op) {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:404: Migration of CUDNN_ATTR_OPERATION_POINTWISE_BDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
                /*
                DPCT1007:405: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.bdesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_BDESC Failed");
                return std::move(m_operation);
            }
        }

        if (m_operation.pointwise_port_count == 4) {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:406: Migration of CUDNN_ATTR_OPERATION_POINTWISE_TDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_POINTWISE_TDESC,
                /*
                DPCT1007:407: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.tdesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_POINTWISE_TDESC Failed");
                return std::move(m_operation);
            }
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_conv_backward_data() {
        m_operation.operationTag = "ConvBwdData";

        auto status = 0;

        auto dxdesc_ = m_operation.dxdesc != nullptr ? m_operation.dxdesc : m_operation.xdesc;
        status       = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:408: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
            /*
            DPCT1007:409: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(dxdesc_->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:410: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
            /*
            DPCT1007:411: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.wdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W Failed");
            return std::move(m_operation);
        }

        auto dydesc_ = m_operation.dydesc != nullptr ? m_operation.dydesc : m_operation.ydesc;
        status       = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:412: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
            /*
            DPCT1007:413: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(dydesc_->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:414: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
            /*
            DPCT1007:415: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.cdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC Failed");
            return std::move(m_operation);
        }

        /*
        DPCT1007:416: Migration of CUDNN_TYPE_FLOAT is not supported.
        */
        void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                     : static_cast<void *>(&m_operation.alpha_d));
        /*
        DPCT1007:417: Migration of CUDNN_TYPE_FLOAT is not supported.
        */
        void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                    : static_cast<void *>(&m_operation.beta_d));
        status     = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:418: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
            m_operation.alphabetaType,
            1,
            alpha);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:419: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
            m_operation.alphabetaType,
            1,
            beta);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        getLogger() << "Extracting the feature vector" << std::endl;
        extract_feature_vector(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_bn_finalize_op() {
        m_operation.operationTag = "BNFinalize";
        auto status              = 0;

        auto set_attribute = [&status](Operation_v8 &operation,
                                       cudnnBackendAttributeName_t attr,
                                       const char *fail_msg,
                                       void const *ptr,
                                       /*
                                       DPCT1007:420: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                       */
                                       cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       int64_t cnt                      = 1) {
            status = cudnn_frontend::set_attribute(operation.pointer->get_backend_descriptor(), attr, type, cnt, ptr);
            if (status != 0) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };

        set_attribute(m_operation,
                      /*
                      DPCT1007:421: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE is not supported.
                      */
                      CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE Failed",
                      &(m_operation.bn_stats_mode),
                      /*
                      DPCT1007:422: Migration of CUDNN_TYPE_BN_FINALIZE_STATS_MODE is not supported.
                      */
                      CUDNN_TYPE_BN_FINALIZE_STATS_MODE,
                      1);
        if (status != 0) {
            return std::move(m_operation);
        }

        set_attribute(m_operation,
                      /*
                      DPCT1007:423: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC is not supported.
                      */
                      CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC Failed",
                      &(m_operation.compute_type),
                      /*
                      DPCT1007:424: Migration of CUDNN_TYPE_DATA_TYPE is not supported.
                      */
                      CUDNN_TYPE_DATA_TYPE,
                      1);
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.sumdesc) {
            set_attribute(m_operation,
                          /*
                          DPCT1007:425: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC Failed",
                          &(m_operation.sumdesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.sqsumdesc) {
            set_attribute(m_operation,
                          /*
                          DPCT1007:426: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC Failed",
                          &(m_operation.sqsumdesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.biasdesc) {
            set_attribute(m_operation,
                          /*
                          DPCT1007:427: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC Failed",
                          &(m_operation.biasdesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.scaledesc) {
            set_attribute(m_operation,
                          /*
                          DPCT1007:428: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC Failed",
                          &(m_operation.scaledesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.eqscaledesc) {
            set_attribute(m_operation,
                          /*
                          DPCT1007:429: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC Failed",
                          &(m_operation.eqscaledesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.eqbiasdesc) {
            set_attribute(m_operation,
                          /*
                          DPCT1007:430: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC Failed",
                          &(m_operation.eqbiasdesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.prevMeandesc) {
            set_attribute(
                m_operation,
                /*
                DPCT1007:431: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC Failed",
                &(m_operation.prevMeandesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.prevVardesc) {
            set_attribute(
                m_operation,
                /*
                DPCT1007:432: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC Failed",
                &(m_operation.prevVardesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.nextMeandesc) {
            set_attribute(
                m_operation,
                /*
                DPCT1007:433: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute "
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC Failed",
                &(m_operation.nextMeandesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.nextVardesc) {
            set_attribute(
                m_operation,
                /*
                DPCT1007:434: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute "
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC Failed",
                &(m_operation.nextVardesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.savedMeandesc) {
            set_attribute(
                m_operation,
                /*
                DPCT1007:435: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC Failed",
                &(m_operation.savedMeandesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.savedInVardesc) {
            set_attribute(
                m_operation,
                /*
                DPCT1007:436: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC Failed",
                &(m_operation.savedInVardesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.epsilondesc) {
            set_attribute(m_operation,
                          /*
                          DPCT1007:437: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC Failed",
                          &(m_operation.epsilondesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.expDecayFactordesc) {
            set_attribute(
                m_operation,
                /*
                DPCT1007:438: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC Failed",
                &(m_operation.expDecayFactordesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        if (m_operation.accumCountdesc) {
            set_attribute(
                m_operation,
                /*
                DPCT1007:439: Migration of CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC Failed",
                &(m_operation.accumCountdesc->get_backend_descriptor()));
            if (status != 0) {
                return std::move(m_operation);
            }
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_genstats_op() {
        m_operation.operationTag = "GenStats";
        auto status              = 0;

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:440: Migration of CUDNN_ATTR_OPERATION_GENSTATS_XDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_GENSTATS_XDESC,
            /*
            DPCT1007:441: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.xdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_XDESC Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:442: Migration of CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC,
            /*
            DPCT1007:443: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.sumdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:444: Migration of CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC,
            /*
            DPCT1007:445: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.sqsumdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:446: Migration of CUDNN_ATTR_OPERATION_GENSTATS_MODE is not supported.
            */
            CUDNN_ATTR_OPERATION_GENSTATS_MODE,
            /*
            DPCT1007:447: Migration of CUDNN_TYPE_GENSTATS_MODE is not supported.
            */
            CUDNN_TYPE_GENSTATS_MODE,
            1,
            &(m_operation.genstats_mode));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_MODE Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:448: Migration of CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC is not supported.
            */
            CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC,
            /*
            DPCT1007:449: Migration of CUDNN_TYPE_DATA_TYPE is not supported.
            */
            CUDNN_TYPE_DATA_TYPE,
            1,
            &(m_operation.compute_type));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }

        return std::move(m_operation);
    }

    Operation_v8 &&
    build_conv_backward_filter() {
        m_operation.operationTag = "ConvBwdFilter";

        auto status = 0;

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:450: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
            /*
            DPCT1007:451: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.xdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X Failed");
            return std::move(m_operation);
        }

        auto dwdesc_ = m_operation.dwdesc != nullptr ? m_operation.dwdesc : m_operation.wdesc;
        status       = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:452: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
            /*
            DPCT1007:453: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(dwdesc_->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW Failed");
            return std::move(m_operation);
        }

        auto dydesc_ = m_operation.dydesc != nullptr ? m_operation.dydesc : m_operation.ydesc;
        status       = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:454: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
            /*
            DPCT1007:455: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(dydesc_->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:456: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
            /*
            DPCT1007:457: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.cdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(&m_operation,
                                          status,
                                          "CUDNN_BACKEND_OPERATION: SetAttribute "
                                          "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC Failed");
            return std::move(m_operation);
        }
        /*
        DPCT1007:458: Migration of CUDNN_TYPE_FLOAT is not supported.
        */
        void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                     : static_cast<void *>(&m_operation.alpha_d));
        /*
        DPCT1007:459: Migration of CUDNN_TYPE_FLOAT is not supported.
        */
        void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                    : static_cast<void *>(&m_operation.beta_d));
        status     = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:460: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
            m_operation.alphabetaType,
            1,
            alpha);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:461: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
            m_operation.alphabetaType,
            1,
            beta);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA Failed");
            return std::move(m_operation);
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        getLogger() << "Extracting the feature vector" << std::endl;
        extract_feature_vector(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_norm_forward() {
        m_operation.operationTag = "Norm_Fwd";
        auto status              = 0;

        auto set_attribute = [&status](Operation_v8 &operation,
                                       cudnnBackendAttributeName_t attr,
                                       const char *fail_msg,
                                       void const *ptr,
                                       /*
                                       DPCT1007:462: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                       */
                                       cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       int64_t cnt                      = 1) {
            status = cudnn_frontend::set_attribute(operation.pointer->get_backend_descriptor(), attr, type, cnt, ptr);
            if (status != 0) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };

        cudnnBackendNormMode_t cudnn_norm_mode;
        status = detail::convert_to_cudnn_type(m_operation.norm_mode, cudnn_norm_mode);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_MODE Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:463: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_MODE is not supported.
            */
            CUDNN_ATTR_OPERATION_NORM_FWD_MODE,
            /*
            DPCT1007:464: Migration of CUDNN_TYPE_NORM_MODE is not supported.
            */
            CUDNN_TYPE_NORM_MODE,
            1,
            &cudnn_norm_mode);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_MODE Failed");
            return std::move(m_operation);
        }

        cudnnBackendNormFwdPhase_t cudnn_norm_fwd_phase;
        status = detail::convert_to_cudnn_type(m_operation.norm_fwd_phase, cudnn_norm_fwd_phase);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_PHASE Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:465: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_PHASE is not supported.
            */
            CUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
            /*
            DPCT1007:466: Migration of CUDNN_TYPE_NORM_FWD_PHASE is not supported.
            */
            CUDNN_TYPE_NORM_FWD_PHASE,
            1,
            &cudnn_norm_fwd_phase);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_PHASE Failed");
            return std::move(m_operation);
        }

        set_attribute(m_operation,
                      /*
                      DPCT1007:467: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_XDESC is not supported.
                      */
                      CUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_XDESC Failed",
                      &m_operation.xdesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.savedMeandesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:468: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC Failed",
                          &m_operation.savedMeandesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.savedInVardesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:469: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC Failed",
                &m_operation.savedInVardesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.scaledesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:470: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC Failed",
                          &m_operation.scaledesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.biasdesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:471: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC Failed",
                          &m_operation.biasdesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.epsilondesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:472: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON Failed",
                          &m_operation.epsilondesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.expDecayFactordesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:473: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR Failed",
                          &m_operation.expDecayFactordesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.prevMeandesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:474: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC Failed",
                &m_operation.prevMeandesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.prevVardesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:475: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC Failed",
                &m_operation.prevVardesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.nextMeandesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:476: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC Failed",
                &m_operation.nextMeandesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.nextVardesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:477: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC Failed",
                &m_operation.nextVardesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.ydesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:478: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_YDESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_FWD_YDESC Failed",
                          &m_operation.ydesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.peerStatdescs.size()) {
            std::vector<cudnnBackendDescriptor_t> backend_peer_stat_descs;
            for (auto &desc : m_operation.peerStatdescs) {
                backend_peer_stat_descs.push_back(desc->get_backend_descriptor());
            }
            set_attribute(
                m_operation,
                /*
                DPCT1007:479: Migration of CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS is not supported.
                */
                CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS Failed",
                backend_peer_stat_descs.data(),
                /*
                DPCT1007:480: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                backend_peer_stat_descs.size());
        }
        if (status != 0) {
            return std::move(m_operation);
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());

        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_norm_backward() {
        m_operation.operationTag = "Norm_Bwd";
        auto status              = 0;

        auto set_attribute = [&status](Operation_v8 &operation,
                                       cudnnBackendAttributeName_t attr,
                                       const char *fail_msg,
                                       void const *ptr,
                                       /*
                                       DPCT1007:481: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                       */
                                       cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       int64_t cnt                      = 1) {
            status = cudnn_frontend::set_attribute(operation.pointer->get_backend_descriptor(), attr, type, cnt, ptr);
            if (status != 0) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };
        cudnnBackendNormMode_t cudnn_norm_mode;
        status = detail::convert_to_cudnn_type(m_operation.norm_mode, cudnn_norm_mode);
        set_attribute(m_operation,
                      /*
                      DPCT1007:482: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_MODE is not supported.
                      */
                      CUDNN_ATTR_OPERATION_NORM_BWD_MODE,
                      "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_MODE Failed",
                      &cudnn_norm_mode,
                      /*
                      DPCT1007:483: Migration of CUDNN_TYPE_NORM_MODE is not supported.
                      */
                      CUDNN_TYPE_NORM_MODE);
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.xdesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:484: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_XDESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_XDESC Failed",
                          &m_operation.xdesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.savedMeandesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:485: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC Failed",
                          &m_operation.savedMeandesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.savedInVardesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:486: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC Failed",
                &m_operation.savedInVardesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.dydesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:487: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC Failed",
                          &m_operation.dydesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.scaledesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:488: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC Failed",
                          &m_operation.scaledesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.dxdesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:489: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC Failed",
                          &m_operation.dxdesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.dscaledesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:490: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC Failed",
                          &m_operation.dscaledesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.dbiasdesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:491: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC Failed",
                          &m_operation.dbiasdesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.peerStatdescs.size()) {
            std::vector<cudnnBackendDescriptor_t> backend_peer_stat_descs;
            for (auto &desc : m_operation.peerStatdescs) {
                backend_peer_stat_descs.push_back(desc->get_backend_descriptor());
            }
            set_attribute(
                m_operation,
                /*
                DPCT1007:492: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS is not supported.
                */
                CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNCUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS Failed",
                backend_peer_stat_descs.data(),
                /*
                DPCT1007:493: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                backend_peer_stat_descs.size());
        }
        if (status != 0) {
            return std::move(m_operation);
        }
        if (m_operation.epsilondesc) {
            set_attribute(m_operation,
                          /*
                          DPCT1007:494: Migration of CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON Failed",
                          &m_operation.epsilondesc->get_backend_descriptor());
        }
        if (status != 0) {
            return std::move(m_operation);
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());

        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_resample_fwd_operation() {
        m_operation.operationTag = "Resample_fwd";
        auto status              = 0;
        status                   = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:495: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
            /*
            DPCT1007:496: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.xdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:497: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
            /*
            DPCT1007:498: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.ydesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:499: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA,
            /*
            DPCT1007:500: Migration of CUDNN_TYPE_DOUBLE is not supported.
            */
            CUDNN_TYPE_DOUBLE,
            1,
            &(m_operation.alpha_d));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:501: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA,
            /*
            DPCT1007:502: Migration of CUDNN_TYPE_DOUBLE is not supported.
            */
            CUDNN_TYPE_DOUBLE,
            1,
            &(m_operation.beta_d));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:503: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC,
            /*
            DPCT1007:504: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.resampledesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC Failed");
            return std::move(m_operation);
        }

        // Maxpooling forward
        if (m_operation.idxdesc != nullptr) {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:505: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                /*
                DPCT1007:506: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.idxdesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC Failed");
                return std::move(m_operation);
            }
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_resample_bwd_operation() {
#if (CUDNN_VERSION >= 8600)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8600, m_operation, "CUDNN_BACKEND_OPERATION: Resample_bwd requires cudnn 8.6.0");
        m_operation.operationTag = "Resample_bwd";
        auto status              = 0;
        status                   = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:507: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC,
            /*
            DPCT1007:508: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.dxdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC Failed");
            return std::move(m_operation);
        }
#if (CUDNN_VERSION >= 8700)
        if (m_operation.xdesc != nullptr) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                8700,
                m_operation,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC requires cudnn 8.7.0");
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:509: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC,
                /*
                DPCT1007:510: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.xdesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC Failed");
                return std::move(m_operation);
            }
        }
        if (m_operation.ydesc != nullptr) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                8700,
                m_operation,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC requires cudnn 8.7.0");
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:511: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC,
                /*
                DPCT1007:512: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.ydesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC Failed");
                return std::move(m_operation);
            }
        }
#endif
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:513: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC,
            /*
            DPCT1007:514: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.dydesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:515: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA,
            /*
            DPCT1007:516: Migration of CUDNN_TYPE_DOUBLE is not supported.
            */
            CUDNN_TYPE_DOUBLE,
            1,
            &(m_operation.alpha_d));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:517: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA,
            /*
            DPCT1007:518: Migration of CUDNN_TYPE_DOUBLE is not supported.
            */
            CUDNN_TYPE_DOUBLE,
            1,
            &(m_operation.beta_d));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:519: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC is not supported.
            */
            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC,
            /*
            DPCT1007:520: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.resampledesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC Failed");
            return std::move(m_operation);
        }

        // Maxpooling backward
        if (m_operation.idxdesc != nullptr) {
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:521: Migration of CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC is not supported.
                */
                CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC,
                /*
                DPCT1007:522: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.idxdesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC Failed");
                return std::move(m_operation);
            }
        }

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Resample operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_rng_operation() {
#if (CUDNN_VERSION >= 8700)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8700, m_operation, "CUDNN_BACKEND_OPERATION: build_rng_operation requires cudnn 8.7.0");
        m_operation.operationTag = "Rng";
        auto status              = 0;
        status =
            cudnn_frontend::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:523: Migration of CUDNN_ATTR_OPERATION_RNG_YDESC is not supported.
                                          */
                                          CUDNN_ATTR_OPERATION_RNG_YDESC,
                                          /*
                                          DPCT1007:524: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &(m_operation.ydesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_YDESC Failed");
            return std::move(m_operation);
        }

#if (CUDNN_VERSION >= 8800)
        // seed can be a tensor or an int64
        // if tensor is defined we give it precedence
        if (m_operation.seeddesc) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                8800,
                m_operation,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_SEED requires cudnn 8.8.0");
            status =
                cudnn_frontend::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                              /*
                                              DPCT1007:525: Migration of CUDNN_ATTR_OPERATION_RNG_SEED is not supported.
                                              */
                                              CUDNN_ATTR_OPERATION_RNG_SEED,
                                              /*
                                              DPCT1007:526: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                              */
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              1,
                                              &(m_operation.seeddesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_SEED Failed");
                return std::move(m_operation);
            }
        } else
#endif
        {
            status =
                cudnn_frontend::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                              /*
                                              DPCT1007:527: Migration of CUDNN_ATTR_OPERATION_RNG_SEED is not supported.
                                              */
                                              CUDNN_ATTR_OPERATION_RNG_SEED,
                                              /*
                                              DPCT1007:528: Migration of CUDNN_TYPE_INT64 is not supported.
                                              */
                                              CUDNN_TYPE_INT64,
                                              1,
                                              &(m_operation.seed));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_SEED Failed");
                return std::move(m_operation);
            }
        }
        status =
            cudnn_frontend::set_attribute(m_operation.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:529: Migration of CUDNN_ATTR_OPERATION_RNG_DESC is not supported.
                                          */
                                          CUDNN_ATTR_OPERATION_RNG_DESC,
                                          /*
                                          DPCT1007:530: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &(m_operation.rngdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_DESC Failed");
            return std::move(m_operation);
        }

#if (CUDNN_VERSION >= 8800)
        if (m_operation.offsetdesc) {
            NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
                8800,
                m_operation,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC requires cudnn 8.8.0");
            status = cudnn_frontend::set_attribute(
                m_operation.pointer->get_backend_descriptor(),
                /*
                DPCT1007:531: Migration of CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC,
                /*
                DPCT1007:532: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                */
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &(m_operation.offsetdesc->get_backend_descriptor()));
            if (status != 0) {
                set_error_and_throw_exception(
                    &m_operation,
                    status,
                    "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC Failed");
                return std::move(m_operation);
            }
        }
#endif

        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Rng operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_reshape_operation() {
#if (CUDNN_VERSION >= 8700)
        NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(
            8700, m_operation, "CUDNN_BACKEND_OPERATION: build_reshape_operation requires cudnn 8.7.0");
        m_operation.operationTag = "Reshape";
        auto status              = 0;
        status                   = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:533: Migration of CUDNN_ATTR_OPERATION_RESHAPE_XDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_RESHAPE_XDESC,
            /*
            DPCT1007:534: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.xdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESHAPE_XDESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:535: Migration of CUDNN_ATTR_OPERATION_RESHAPE_YDESC is not supported.
            */
            CUDNN_ATTR_OPERATION_RESHAPE_YDESC,
            /*
            DPCT1007:536: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.ydesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_RESHAPE_YDESC Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
#else
        set_error_and_throw_exception(&m_operation,
                                      CUDNN_STATUS_NOT_SUPPORTED,
                                      "CUDNN_BACKEND_OPERATION: Reshape operation Not supported in this version");
#endif
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_bn_bwd_weight_op() {
        m_operation.operationTag = "Dgrad_Drelu_BN_Bwd";
        auto status              = 0;

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:537: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC is not supported.
            */
            CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC,
            /*
            DPCT1007:538: Migration of CUDNN_TYPE_DATA_TYPE is not supported.
            */
            CUDNN_TYPE_DATA_TYPE,
            1,
            &(m_operation.compute_type));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC Failed");
            return std::move(m_operation);
        }

        auto set_attribute = [&status](Operation_v8 &operation,
                                       cudnnBackendAttributeName_t attr,
                                       const char *fail_msg,
                                       void const *ptr,
                                       /*
                                       DPCT1007:539: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                       */
                                       cudnnBackendAttributeType_t type = CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       int64_t cnt                      = 1) {
            status = cudnn_frontend::set_attribute(operation.pointer->get_backend_descriptor(), attr, type, cnt, ptr);
            if (status != 0) {
                set_error_and_throw_exception(&operation, status, fail_msg);
            }
        };

        if (m_operation.xdesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:540: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC Failed",
                          &m_operation.xdesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.savedMeandesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:541: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC Failed",
                          &m_operation.savedMeandesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.savedInVardesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:542: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC Failed",
                &m_operation.savedInVardesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.scaledesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:543: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC Failed",
                &m_operation.scaledesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.dydesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:544: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC Failed",
                          &m_operation.dydesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.dscaledesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:545: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC Failed",
                &m_operation.dscaledesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.dbiasdesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:546: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC Failed",
                &m_operation.dbiasdesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.eqscaledesc)
            set_attribute(
                m_operation,
                /*
                DPCT1007:547: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC Failed",
                &m_operation.eqscaledesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.eqscaledesc1)
            set_attribute(
                m_operation,
                /*
                DPCT1007:548: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC is not supported.
                */
                CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC Failed",
                &m_operation.eqscaledesc1->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }

        if (m_operation.eqbiasdesc)
            set_attribute(m_operation,
                          /*
                          DPCT1007:549: Migration of CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS is not supported.
                          */
                          CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS,
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS Failed",
                          &m_operation.eqbiasdesc->get_backend_descriptor());
        if (status != 0) {
            return std::move(m_operation);
        }
        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }
        return std::move(m_operation);
    }

    Operation_v8 &&
    build_conv_forward() {
        m_operation.operationTag = "ConvFwd";

        auto status = 0;

        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:550: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
            /*
            DPCT1007:551: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.xdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:552: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
            /*
            DPCT1007:553: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.wdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:554: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
            /*
            DPCT1007:555: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.ydesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:556: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
            /*
            DPCT1007:557: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_operation.cdesc->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC Failed");
            return std::move(m_operation);
        }
        /*
        DPCT1007:558: Migration of CUDNN_TYPE_FLOAT is not supported.
        */
        void *alpha = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.alpha_s)
                                                                     : static_cast<void *>(&m_operation.alpha_d));
        /*
        DPCT1007:559: Migration of CUDNN_TYPE_FLOAT is not supported.
        */
        void *beta = (m_operation.alphabetaType == CUDNN_TYPE_FLOAT ? static_cast<void *>(&m_operation.beta_s)
                                                                    : static_cast<void *>(&m_operation.beta_d));
        status     = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:560: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
            m_operation.alphabetaType,
            1,
            alpha);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::set_attribute(
            m_operation.pointer->get_backend_descriptor(),
            /*
            DPCT1007:561: Migration of CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA is not supported.
            */
            CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
            m_operation.alphabetaType,
            1,
            beta);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA Failed");
            return std::move(m_operation);
        }
        status = cudnn_frontend::finalize(m_operation.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");
            return std::move(m_operation);
        }

        getLogger() << "Extracting the feature vector" << std::endl;
        extract_feature_vector(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        return std::move(m_operation);
    }

    void
    extract_feature_vector(DescriptorType_t op_type) {
        /// Build the feature vector of this operation now.
        m_operation.feature_vector.reserve(50);

        m_operation.feature_vector.push_back(static_cast<int>(op_type));
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(xTensor_dimA[i]);  // n, c, (g), d, h , w
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(wTensor_dimA[i]);  // n, c, (g), d, h , w
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(yTensor_dimA[i]);  // n, c, (g), d, h , w
        }
        const int max_spatial_dim = 3;

        /// Padding
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == max_spatial_dim - 1 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_padding[i]);
            }
        }
        /// Dilation
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == max_spatial_dim - 1 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_dilation[i]);
            }
        }
        /// Strides
        for (auto i = 0; i < max_spatial_dim; i++) {
            if (i == max_spatial_dim - 1 && is2D) {
                m_operation.feature_vector.push_back(0);
            } else {
                m_operation.feature_vector.push_back(conv_stride[i]);
            }
        }

        m_operation.feature_vector.push_back(xType);
        m_operation.feature_vector.push_back(wType);
        m_operation.feature_vector.push_back(yType);
        m_operation.feature_vector.push_back(cType);
        m_operation.feature_vector.push_back(mode);

        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(xTensor_strA[i]);  // n, c, (g), d, h , w
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(wTensor_strA[i]);  // n, c, (g), d, h , w
        }
        for (auto i = 0; i < tensor_dims; i++) {
            m_operation.feature_vector.push_back(yTensor_strA[i]);  // n, c, (g), d, h , w
        }

        int64_t alpha_as_int;
        int64_t beta_as_int;
        std::memcpy((void *)&alpha_as_int, (void *)(&m_operation.alpha_s), sizeof(int64_t));
        std::memcpy((void *)&beta_as_int, (void *)(&m_operation.beta_s), sizeof(int64_t));

        m_operation.feature_vector.push_back(alpha_as_int);
        m_operation.feature_vector.push_back(beta_as_int);
    }

    dpct::err1
    validate_matmul_op(Message_t &msg) {
        if (m_operation.matmuldesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_DESC";
            return 2000;
        }
        if (m_operation.amatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_ADESC";
            return 2000;
        }
        if (m_operation.bmatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_BDESC";
            return 2000;
        }
        if (m_operation.cmatdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_MATMUL_CDESC";
            return 2000;
        }
        return 0;
    }

    dpct::err1
    validate_norm_op(Message_t &msg) {
        dpct::err1 status = 0;
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_NORM.*XDESC";
            return 2000;
        }

        if (get_backend_version() == 8500) {
            std::array<int64_t, 10> x_dimensions;
            int64_t dim_count;
            status =
                cudnn_frontend::get_attribute(m_operation.xdesc->get_backend_descriptor(),
                                              /*
                                              DPCT1007:562: Migration of CUDNN_ATTR_TENSOR_DIMENSIONS is not supported.
                                              */
                                              CUDNN_ATTR_TENSOR_DIMENSIONS,
                                              /*
                                              DPCT1007:563: Migration of CUDNN_TYPE_INT64 is not supported.
                                              */
                                              CUDNN_TYPE_INT64,
                                              x_dimensions.size(),
                                              &dim_count,
                                              x_dimensions.data());
            if (status != 0) {
                msg = "CUDNN_BACKEND_OPERATION: CUDNN_BACKEND_TENSOR has invalid CUDNN_ATTR_TENSOR_DIMENSIONS";
                return status;
            }

            int64_t N = x_dimensions[0];
            int64_t C = x_dimensions[1];

            if ((N != 1) || ((C % 8) != 0)) {
                msg = "CUDNN_BACKEND_OPERATION: CUDNN_BACKEND_TENSOR has bad CUDNN_ATTR_TENSOR_DIMENSIONS";
                return 2000;
            }
        }

        return status;
    }

    dpct::err1
    validate_resample_op(Message_t &msg) {
        if (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR) {
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*XDESC";
                return 2000;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*YDESC";
                return 2000;
            }
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR) {
            if (m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*DXDESC";
                return 2000;
            }
            if (m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*DYDESC";
                return 2000;
            }
        }

        if (m_operation.resampledesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESAMPLE.*RESAMPLEDESC";
            return 2000;
        }

        return 0;
    }

    dpct::err1
    validate_rng_op(Message_t &msg) {
        if (m_operation.ydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_YDESC";
            return 2000;
        }

        if (m_operation.rngdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RNG_DESC";
            return 2000;
        }

        return 0;
    }

    dpct::err1
    validate_reshape_op(Message_t &msg) {
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESHAPE_XDESC";
            return 2000;
        }

        if (m_operation.ydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_RESHAPE_YDESC";
            return 2000;
        }

        return 0;
    }

    dpct::err1
    validate_bn_bwd_weight_op(Message_t &msg) {
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC";
            return 2000;
        }

        if (m_operation.dydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC";
            return 2000;
        }

        if (m_operation.savedMeandesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC";
            return 2000;
        }

        if (m_operation.savedInVardesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC";
            return 2000;
        }

        return 0;
    }

    dpct::err1
    validate_reduction_op(Message_t &msg) {
        if (m_operation.reductiondesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_DESC";
            return 2000;
        }
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_XDESC";
            return 2000;
        }
        if (m_operation.ydesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_REDUCTION_YDESC";
            return 2000;
        }
        return 0;
    }

    dpct::err1
    validate_pointwise_op(Message_t &msg) {
        if (m_operation.xdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_XDESC";
            return 2000;
        }
        if (m_operation.is_pointwise_math_op) {
            if (m_operation.pointwise_port_count == 3 && m_operation.bdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_BDESC";
                return 2000;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_YDESC";
                return 2000;
            }
        } else if (m_operation.is_pointwise_activation_fwd_op || m_operation.is_pointwise_identity_op) {
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_YDESC";
                return 2000;
            }
        } else if (m_operation.is_pointwise_activation_bwd_op) {
            if (m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_DYDESC";
                return 2000;
            }
            if (m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_POINTWISE_DXDESC";
                return 2000;
            }
        } else {
            msg = "CUDNN_BACKEND_OPERATION: Unsupported cudnn pointwise mode. Check PointwiseMode_t::*";
            return 2000;
        }
        return 0;
    }

    dpct::err1
    validate_convolution_op(Message_t &msg) {
        if (m_operation.cdesc == nullptr) {
            msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_CONV_DESC";
            return 2000;
        }
        if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X";
                return 2000;
            }
            if (m_operation.wdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W";
                return 2000;
            }
            if (m_operation.ydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_Y";
                return 2000;
            }

        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            if (m_operation.ydesc != nullptr && m_operation.dydesc != nullptr) {
                msg =
                    "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setyDesc() or "
                    "setdyDesc()";
                return 2000;
            }
            if (m_operation.ydesc == nullptr && m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setyDesc() or setdyDesc()";
                return 2000;
            }
            if (m_operation.xdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_X";
                return 2000;
            }
            if (m_operation.wdesc != nullptr && m_operation.dwdesc != nullptr) {
                msg =
                    "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setwDesc() or "
                    "setdwDesc()";
                return 2000;
            }
            if (m_operation.wdesc == nullptr && m_operation.dwdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setwDesc() or setdwDesc()";
                return 2000;
            }
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            if (m_operation.ydesc != nullptr && m_operation.dydesc != nullptr) {
                msg =
                    "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setyDesc() or "
                    "setdyDesc()";
                return 2000;
            }
            if (m_operation.ydesc == nullptr && m_operation.dydesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setyDesc() or setdyDesc()";
                return 2000;
            }
            if (m_operation.wdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Check and Set the CUDNN_ATTR_OPERATION_CONVOLUTION_*_W";
                return 2000;
            }
            if (m_operation.xdesc != nullptr && m_operation.dxdesc != nullptr) {
                msg =
                    "CUDNN_BACKEND_OPERATION: Ambiguous specification. Choose and Set only one of setxDesc() or "
                    "setdxDesc()";
                return 2000;
            }
            if (m_operation.xdesc == nullptr && m_operation.dxdesc == nullptr) {
                msg = "CUDNN_BACKEND_OPERATION: Choose and Set one of setxDesc() or setdxDesc()";
                return 2000;
            }
        } else {
            msg =
                "CUDNN_BACKEND_OPERATION: Unsupported convolution operation. Check and set "
                "CUDNN_BACKEND_OPERATION_CONVOLUTION_*_DESCRIPTOR";
            return 2000;
        }
        return 0;
    }

    void
    copy_dims_and_strides(const int64_t *from, int64_t *to) const {
        for (auto i = 0; i < CUDNN_DIM_MAX + 1; i++) {
            to[i] = from[i];
        }
    }

   public:
    /** @defgroup OperationBuilder_v8
     *  Set individual property of Operation_v8 class
     *  @{
     */
    /// Will be Deprecated Do not use
    auto
    setxDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.xdesc = raw_tensor;
        return *this;
    }

    auto
    setxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.xdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), xTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), xTensor_strA);
        tensor_dims = tensor.getDimensionCount();
        xType       = tensor.getDataType();
        return *this;
    }
    auto
    setbDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need bTensor");
        }
        m_operation.bdesc = tensor.get_desc();
        return *this;
    }

    auto
    settDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need tTensor");
        }
        m_operation.tdesc = tensor.get_desc();
        return *this;
    }

    auto
    setyDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.ydesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), yTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), yTensor_strA);
        yType = tensor.getDataType();
        return *this;
    }
    auto
    setwDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_convolution_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Convolution operation does not need wTensor");
        }
        m_operation.wdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), wTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), wTensor_strA);
        wType = tensor.getDataType();
        return *this;
    }

    /// Will be Deprecated Do not use
    auto
    setdyDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.dydesc = raw_tensor;
        return *this;
    }
    auto
    setdyDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dydesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), yTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), yTensor_strA);
        yType = tensor.getDataType();
        return *this;
    }
    auto
    setdxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dxdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), xTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), xTensor_strA);
        tensor_dims = tensor.getDimensionCount();
        xType       = tensor.getDataType();
        return *this;
    }
    auto
    setdwDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.dwdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), wTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), wTensor_strA);
        wType = tensor.getDataType();
        return *this;
    }
    auto
    setResampleDesc(ResampleDesc_v8 const &resampleDesc) -> OperationBuilder_v8 & {
        if (is_resample_fwd_op == false && is_resample_bwd_op == false) {
            set_error_and_throw_exception(
                &m_operation, 2000, "RESAMPLE_DESC: Non Resample operation does not need Resample DESCRIPTOR");
        }
        m_operation.resampledesc = resampleDesc.get_desc();
        return *this;
    }

    auto
    setRngDesc(RngDesc_v8 const &rngDesc) -> OperationBuilder_v8 & {
        if (is_rng_op == false) {
            set_error_and_throw_exception(
                &m_operation, 2000, "RNG_DESC: Non Rng operation does not need Rng DESCRIPTOR");
        }
        m_operation.rngdesc = rngDesc.get_desc();
        return *this;
    }

    auto
    setidxDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.idxdesc = tensor.get_desc();
        copy_dims_and_strides(tensor.getDimArray(), idxTensor_dimA);
        copy_dims_and_strides(tensor.getStrideArray(), idxTensor_strA);
        idxType = tensor.getDataType();
        return *this;
    }

    auto
    setSeedDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.seeddesc = tensor.get_desc();
        return *this;
    }

    auto
    setOffsetDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.offsetdesc = tensor.get_desc();
        return *this;
    }

    auto
    setcDesc(ConvDesc_v8 const &conv) -> OperationBuilder_v8 & {
        if (is_convolution_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Convolution operation does not need Convolution DESCRIPTOR");
        }
        m_operation.cdesc = conv.get_desc();
        if (conv.getComputePrecision() == DataType_t::DOUBLE) {
            /*
            DPCT1007:564: Migration of CUDNN_TYPE_DOUBLE is not supported.
            */
            m_operation.alphabetaType = CUDNN_TYPE_DOUBLE;
        }
        is2D = conv.getDimensionCount() == 2;
        copy_dims_and_strides(conv.getPadding(), conv_padding);
        copy_dims_and_strides(conv.getDilation(), conv_dilation);
        copy_dims_and_strides(conv.getStride(), conv_stride);
        cType = static_cast<int>(conv.getComputePrecision());
        mode  = conv.getMathMode();
        return *this;
    }

    auto
    setNormFwdPhase(NormFwdPhase_t mode) -> OperationBuilder_v8 & {
        m_operation.norm_fwd_phase = mode;
        return *this;
    }

    auto
    setNormalizationMode(NormMode_t mode) -> OperationBuilder_v8 & {
        m_operation.norm_mode = mode;
        return *this;
    }

    // To be deprecated. Please use setNormalizationMode(cudnn_frontend::NormMode_t mode) instead.
    auto
    setNormalizationMode(cudnnBackendNormMode_t mode) -> OperationBuilder_v8 & {
        detail::convert_from_cudnn_type(mode, m_operation.norm_mode);
        return *this;
    }

    // To be deprecated. Please use setNormFwdPhase(cudnn_frontend::NormFwdPhase_t mode) instead.
    auto
    setNormFwdPhase(cudnnBackendNormFwdPhase_t mode) -> OperationBuilder_v8 & {
        detail::convert_from_cudnn_type(mode, m_operation.norm_fwd_phase);
        return *this;
    }

    auto
    setBNFinalizeMode(cudnnBnFinalizeStatsMode_t mode) -> OperationBuilder_v8 & {
        m_operation.bn_stats_mode = mode;
        return *this;
    }

    auto
    setAccumCountTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.accumCountdesc = tensor.get_desc();
        return *this;
    }

    auto
    setEpsilonTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.epsilondesc = tensor.get_desc();
        return *this;
    }

    auto
    setExpDecayFactorTensor(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.expDecayFactordesc = tensor.get_desc();
        return *this;
    }

    auto
    addPeerStatTensor(Tensor_v8 const &peer_stat_tensor) -> OperationBuilder_v8 & {
        m_operation.peerStatdescs.push_back(peer_stat_tensor.get_desc());
        return *this;
    }

    auto
    setPeerStatTensor(std::vector<Tensor_v8> const &peer_stat_tensors) -> OperationBuilder_v8 & {
        for (auto &tensor : peer_stat_tensors) {
            m_operation.peerStatdescs.push_back(tensor.get_desc());
        }
        return *this;
    }

    auto
    setPrevRunningMeanAndVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.prevMeandesc = mean.get_desc();
        m_operation.prevVardesc  = var.get_desc();
        return *this;
    }

    auto
    setNextRunningMeanAndVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.nextMeandesc = mean.get_desc();
        m_operation.nextVardesc  = var.get_desc();
        return *this;
    }

    auto
    setSavedMeanAndInvVar(Tensor_v8 const &mean, Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.savedMeandesc  = mean.get_desc();
        m_operation.savedInVardesc = var.get_desc();
        return *this;
    }

    auto
    setSavedInvVar(Tensor_v8 const &var) -> OperationBuilder_v8 & {
        m_operation.savedInVardesc = var.get_desc();
        return *this;
    }

    auto
    setScale(Tensor_v8 const &scale_tensor) -> OperationBuilder_v8 & {
        m_operation.scaledesc = scale_tensor.get_desc();
        return *this;
    }

    auto
    setBias(Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.biasdesc = bias_tensor.get_desc();
        return *this;
    }

    auto
    setScaleAndBias(Tensor_v8 const &scale_tensor, Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.scaledesc = scale_tensor.get_desc();
        m_operation.biasdesc  = bias_tensor.get_desc();
        return *this;
    }

    auto
    setDScale(Tensor_v8 const &scale_tensor) -> OperationBuilder_v8 & {
        m_operation.dscaledesc = scale_tensor.get_desc();
        return *this;
    }

    auto
    setDBias(Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.dbiasdesc = bias_tensor.get_desc();
        return *this;
    }

    auto
    setDScaleAndDBias(Tensor_v8 const &scale_tensor, Tensor_v8 const &bias_tensor) -> OperationBuilder_v8 & {
        m_operation.dscaledesc = scale_tensor.get_desc();
        m_operation.dbiasdesc  = bias_tensor.get_desc();
        return *this;
    }

    auto
    setEqScalesAndBias(Tensor_v8 const &eq_scale_tensor1,
                       Tensor_v8 const &eq_scale_tensor2,
                       Tensor_v8 const &eq_bias_tensor) -> OperationBuilder_v8 & {
        m_operation.eqscaledesc  = eq_scale_tensor1.get_desc();
        m_operation.eqscaledesc1 = eq_scale_tensor2.get_desc();
        m_operation.eqbiasdesc   = eq_bias_tensor.get_desc();
        return *this;
    }

    auto
    setEqScaleAndBias(Tensor_v8 const &eq_scale_tensor, Tensor_v8 const &eq_bias_tensor) -> OperationBuilder_v8 & {
        m_operation.eqscaledesc = eq_scale_tensor.get_desc();
        m_operation.eqbiasdesc  = eq_bias_tensor.get_desc();
        return *this;
    }

    auto
    setSumDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.sumdesc = tensor.get_desc();
        return *this;
    }

    auto
    setSqSumDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.sqsumdesc = tensor.get_desc();
        return *this;
    }

    auto
    setaMatDesc(ManagedOpaqueDescriptor const &raw_tensor) -> OperationBuilder_v8 & {
        m_operation.amatdesc = raw_tensor;
        return *this;
    }
    auto
    setaMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need a Matrix Tensor");
        }
        m_operation.amatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setbMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need b Matrix Tensor");
        }
        m_operation.bmatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setcMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need c Matrix Tensor");
        }
        m_operation.cmatdesc = tensor.get_desc();
        return *this;
    }
    auto
    setmOverrideDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need mOverride Tensor");
        }
        m_operation.moverridedesc = tensor.get_desc();
        return *this;
    }
    auto
    setnOverrideDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need nOverride Tensor");
        }
        m_operation.noverridedesc = tensor.get_desc();
        return *this;
    }
    auto
    setkOverrideDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need kOverride Tensor");
        }
        m_operation.koverridedesc = tensor.get_desc();
        return *this;
    }
    auto
    setmatmulDesc(MatMulDesc_v8 const &matmulDesc) -> OperationBuilder_v8 & {
        if (is_matmul_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Matmul operation does not need MATMUL DESCRIPTOR");
        }
        m_operation.matmuldesc = matmulDesc.get_desc();
        return *this;
    }
    auto
    setreductionDesc(ReductionDesc_v8 const &reductionDesc) -> OperationBuilder_v8 & {
        if (is_reduction_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Reduction operation does not need REDUCTION DESCRIPTOR");
        }
        m_operation.reductiondesc = reductionDesc.get_desc();
        return *this;
    }
    auto
    setpwDesc(PointWiseDesc_v8 const &pointWiseDesc) -> OperationBuilder_v8 & {
        if (is_pointwise_op == false) {
            set_error_and_throw_exception(
                &m_operation,
                2000,
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR: Non Pointwise operation does not need POINTWISE DESCRIPTOR");
        }
        m_operation.pwdesc               = pointWiseDesc.get_desc();
        m_operation.pointwise_port_count = pointWiseDesc.getPortCount();
        m_operation.pointwise_mode       = pointWiseDesc.getPointWiseMode();

        m_operation.is_pointwise_math_op = ((m_operation.pointwise_mode == PointwiseMode_t::ADD) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::MUL) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::DIV) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::SUB) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::ADD_SQUARE) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::RSQRT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::SIN) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::COS) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::TAN) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::LOGICAL_OR) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::LOGICAL_AND) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::LOGICAL_NOT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_EQ) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_NEQ) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_GT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_GE) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_LT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CMP_LE) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::LOG) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::NEG) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::MOD) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::POW) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::ABS) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::CEIL) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::FLOOR) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::GEN_INDEX) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::BINARY_SELECT) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::ERF) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::RECIPROCAL) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::MIN) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::MAX) ||
                                            (m_operation.pointwise_mode == PointwiseMode_t::SQRT));

        m_operation.is_pointwise_identity_op = (m_operation.pointwise_mode == PointwiseMode_t::IDENTITY);

        m_operation.is_pointwise_activation_fwd_op =
            ((m_operation.pointwise_mode == PointwiseMode_t::RELU_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::TANH_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SIGMOID_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::ELU_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::GELU_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::GELU_APPROX_TANH_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SOFTPLUS_FWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::EXP) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SWISH_FWD));

        m_operation.is_pointwise_activation_bwd_op =
            ((m_operation.pointwise_mode == PointwiseMode_t::RELU_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::TANH_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SIGMOID_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::ELU_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::GELU_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::GELU_APPROX_TANH_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SOFTPLUS_BWD) ||
             (m_operation.pointwise_mode == PointwiseMode_t::SWISH_BWD));

        return *this;
    }

    auto
    setAlpha(float alpha) -> OperationBuilder_v8 & {
        m_operation.alpha_d = static_cast<double>(alpha);
        m_operation.alpha_s = alpha;
        return *this;
    }
    auto
    setAlpha(double alpha) -> OperationBuilder_v8 & {
        m_operation.alpha_s = static_cast<float>(alpha);
        m_operation.alpha_d = alpha;
        return *this;
    }
    auto
    setAlpha2(float alpha) -> OperationBuilder_v8 & {
        m_operation.alpha2_d = static_cast<double>(alpha);
        m_operation.alpha2_s = alpha;
        return *this;
    }
    auto
    setAlpha2(double alpha) -> OperationBuilder_v8 & {
        m_operation.alpha2_s = static_cast<float>(alpha);
        m_operation.alpha2_d = alpha;
        return *this;
    }
    auto
    setBeta(float beta) -> OperationBuilder_v8 & {
        m_operation.beta_d = static_cast<double>(beta);
        m_operation.beta_s = beta;
        return *this;
    }
    auto
    setBeta(double beta) -> OperationBuilder_v8 & {
        m_operation.beta_s = static_cast<float>(beta);
        m_operation.beta_d = beta;
        return *this;
    }

    auto
    setSeed(int64_t seed) -> OperationBuilder_v8 & {
        m_operation.seed = seed;
        return *this;
    }

    auto
    setComputeType(dpct::library_data_t dtype) -> OperationBuilder_v8 & {
        m_operation.compute_type = dtype;
        return *this;
    }

    auto
    setMathPrecision(dpct::library_data_t dtype) -> OperationBuilder_v8 & {
        return setComputeType(dtype);
    }

    auto
    setGenStatsMode(cudnnGenStatsMode_t type) -> OperationBuilder_v8 & {
        m_operation.genstats_mode = type;
        return *this;
    }

    OperationBuilder_v8(DescriptorType_t mode) {
        m_operation.op_mode = mode;
        is_convolution_op =
            ((m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) ||
             (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) ||
             (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR));

        is_pointwise_op     = (m_operation.op_mode == DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
        is_matmul_op        = (m_operation.op_mode == DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR);
        is_reduction_op     = (m_operation.op_mode == DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR);
        is_genstats_op      = (m_operation.op_mode == DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR);
        is_bn_finalize_op   = (m_operation.op_mode == DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR);
        is_bn_bwd_weight    = (m_operation.op_mode == DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR);
        is_resample_fwd_op  = (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR);
        is_norm_forward_op  = (m_operation.op_mode == DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);
        is_norm_backward_op = (m_operation.op_mode == DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR);
        is_resample_bwd_op  = (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR);
        is_rng_op           = (m_operation.op_mode == DescriptorType_t::OPERATION_RNG_DESCRIPTOR);
        is_reshape_op       = (m_operation.op_mode == DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR);
    }

    // This constructor which takes in cudnn C backend enum for cudnnBackendDescriptorType_t will be deprecated,
    // in favour of OperationBuilder_v8(cudnn_frontend::DescriptorType_t)
    OperationBuilder_v8(cudnnBackendDescriptorType_t mode)
        : OperationBuilder_v8(detail::convert_from_cudnn_type(mode)) {}

    /** @} */

    //! constructs the backend Operation_v8 by calling the cudnn API
    //! Throws the appropriate error message
    Operation_v8 &&
    build() {
        if (m_operation.status != 0) {
            set_error_and_throw_exception(
                &m_operation, m_operation.status, "CUDNN_BACKEND_OPERATION: Operation not initialized properly");
            return std::move(m_operation);
        }

        Message_t msg         = nullptr;
        dpct::err1 status_    = 0;
        if (is_convolution_op) {
            status_ = validate_convolution_op(msg);
        } else if (is_pointwise_op) {
            status_ = validate_pointwise_op(msg);
        } else if (is_matmul_op) {
            status_ = validate_matmul_op(msg);
        } else if (is_reduction_op) {
            status_ = validate_reduction_op(msg);
        } else if (is_genstats_op) {
            status_ = 0;
        } else if (is_bn_finalize_op) {
            status_ = 0;
        } else if (is_bn_bwd_weight) {
            status_ = validate_bn_bwd_weight_op(msg);
        } else if (is_resample_fwd_op) {
            status_ = validate_resample_op(msg);
        } else if (is_resample_bwd_op) {
            status_ = validate_resample_op(msg);
        } else if (is_rng_op) {
            status_ = validate_rng_op(msg);
        } else if (is_norm_forward_op || is_norm_backward_op) {
            status_ = validate_norm_op(msg);
        } else if (is_reshape_op) {
            status_ = validate_reshape_op(msg);
        } else {
            status_ = 2000;
            msg =
                "CUDNN_BACKEND_OPERATION_DESCRIPTOR: Unsupported cudnn backend descriptor type. Check and set "
                "CUDNN_BACKEND_OPERATION_*_DESCRIPTOR";
        }
        if (status_ != 0) {
            set_error_and_throw_exception(&m_operation, status_, msg);
            return std::move(m_operation);
        }

        // Create the descriptor.
        cudnnBackendDescriptorType_t cudnn_backend_descritpor_type;
        auto status = detail::convert_to_cudnn_type(m_operation.op_mode, cudnn_backend_descritpor_type);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operation,
                status,
                "CUDNN_BACKEND_OPERATION: cudnnCreate Failed with Invalid backend descriptor type.");
            return std::move(m_operation);
        }
        status = m_operation.initialize_managed_backend_pointer(cudnn_backend_descritpor_type);
        if (status != 0) {
            set_error_and_throw_exception(&m_operation, status, "CUDNN_BACKEND_OPERATION: cudnnCreate Failed");
            return std::move(m_operation);
        }

        if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            return build_conv_forward();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            return build_conv_backward_filter();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            return build_conv_backward_data();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR) {
            return build_pointwise_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR) {
            return build_matmul_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR) {
            return build_reduction_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR) {
            return build_genstats_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR) {
            return build_bn_finalize_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR) {
            return build_bn_bwd_weight_op();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR) {
            return build_resample_fwd_operation();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR) {
            return build_norm_forward();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR) {
            return build_norm_backward();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR) {
            return build_resample_bwd_operation();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RNG_DESCRIPTOR) {
            return build_rng_operation();
        } else if (m_operation.op_mode == DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR) {
            return build_reshape_operation();
        } else {
            set_error_and_throw_exception(
                &m_operation, status, "CUDNN_BACKEND_OPERATION: unimplemented operation in frontend");
        }
        getLogger() << "[cudnn_frontend] " << m_operation << std::endl;
        return std::move(m_operation);
    }
};

using Operation        = Operation_v8;
using OperationBuilder = OperationBuilder_v8;
}  // namespace cudnn_frontend
