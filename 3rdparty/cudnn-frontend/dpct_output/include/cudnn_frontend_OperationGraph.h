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
#include <functional>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_utils.h"

// Compile time constant for max ops in a op graph
constexpr int64_t MAX_OPGRAPH_OPS = 50;

namespace cudnn_frontend {

///
/// OperationGraph_v8 Class
/// This class tells the properties of the Tensor_v8 on which the operation will be
/// performed
/// Properties:
///    - handle
///    - operation
///
/// Use OperationGraphBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class OperationGraph_v8 : public BackendDescriptor {
   public:
    friend class OperationGraphBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR has " << numOps << "operations." << std::endl;
        ss << "Tag: " << opGraphTag << std::endl;
        return ss.str();
    }

    OperationGraph_v8(OperationGraph_v8 &&from) = default;
    OperationGraph_v8 &
    operator=(OperationGraph_v8 &&from) = default;

    ~OperationGraph_v8() = default;

    /** @defgroup OperationGraphQuery
     *  Query individual property of OperationGraph_v8 class
     *  @{
     */
    //! Query the total count of the engines for the Operation Set
    auto
    getEngineCount(void) const -> int64_t {
        int64_t global_count = -1;
        auto status          = cudnn_frontend::get_attribute(
            pointer->get_backend_descriptor(),
            /*
            DPCT1007:565: Migration of CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT is not supported.
            */
            CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
            /*
            DPCT1007:566: Migration of CUDNN_TYPE_INT64 is not supported.
            */
            CUDNN_TYPE_INT64,
            1,
            nullptr,
            &global_count);
        if (status != 0) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT Failed");
        }
        return global_count;
    }
    /** @} */

    uint64_t
    getOpCount() const {
        return numOps;
    }

    std::string const &
    getTag() const {
        return opGraphTag;
    }

    bool
    setFeatureVector(feature_vector_t fv) {
        feature_vectors.push_back(fv);
        return true;
    }

    feature_vector_t
    getFeatureVector() const {
        if (feature_vectors.size() != 0) {
            return feature_vectors[0];
        } else {
            return {};
        }
    }

    const std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> &
    getOps() const {
        return ops;
    }

   private:
    OperationGraph_v8()                          = default;
    OperationGraph_v8(OperationGraph_v8 const &) = delete;
    OperationGraph_v8 &
    operator=(OperationGraph_v8 const &) = delete;

    dpct::dnnl::engine_ext handle = nullptr;
    std::array<ManagedOpaqueDescriptor, MAX_OPGRAPH_OPS> ops{};
    int64_t numOps         = -1;
    std::string opGraphTag = "";
    std::vector<feature_vector_t> feature_vectors;
};

///
/// OperationGraphBuilder_v8 Class
/// Helper class used to build OperationGraph_v8 class
class OperationGraphBuilder_v8 {
   public:
    /** @defgroup OperationGraphBuilder_v8
     *  Set individual property of OperationGraph_v8 class
     *  @{
     */
    //! Set cudnnHandle for the operations
    auto
    setHandle(dpct::dnnl::engine_ext handle_) -> OperationGraphBuilder_v8 & {
        m_operationGraph.handle = handle_;
        return *this;
    }
    //! Set numoperations and the operations
    auto
    setOperationGraph(int64_t numOps_, Operation_v8 const **ops_) -> OperationGraphBuilder_v8 & {
        m_operationGraph.numOps = numOps_;
        m_operationGraph.feature_vectors.resize(static_cast<size_t>(numOps_));
        for (auto i = 0u; i < numOps_; i++) {
            m_operationGraph.ops[i] = ops_[i]->get_desc();
            m_operationGraph.opGraphTag += ops_[i]->getTag() + '_';
            m_operationGraph.feature_vectors[i] = ops_[i]->getFeatureVector();
        }
        return *this;
    }

    //! Set numoperations and the operations
    auto
    setOperationGraph(std::vector<Operation> const &ops_) -> OperationGraphBuilder_v8 & {
        m_operationGraph.numOps = ops_.size();
        m_operationGraph.feature_vectors.resize(ops_.size());
        for (auto i = 0u; i < ops_.size(); i++) {
            m_operationGraph.ops[i] = ops_[i].get_desc();
            m_operationGraph.opGraphTag += ops_[i].getTag() + '_';
            m_operationGraph.feature_vectors[i] = ops_[i].getFeatureVector();
        }
        return *this;
    }
    /** @} */

    //! constructs the OperationGraph_v8 by calling the cudnn API
    //! Throws the appropriate error message
    OperationGraph_v8 &&
    build() {
        if (m_operationGraph.numOps <= 0) {
            set_error_and_throw_exception(
                &m_operationGraph,
                2000,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and Set the CUDNN_ATTR_OPERATIONGRAPH_OPS Count field");
            return std::move(m_operationGraph);
        }
        if (m_operationGraph.ops[0] == nullptr) {
            set_error_and_throw_exception(
                &m_operationGraph,
                2000,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and set CUDNN_ATTR_OPERATIONGRAPH_OPS field");
            return std::move(m_operationGraph);
        }
        if (m_operationGraph.handle == nullptr) {
            set_error_and_throw_exception(
                &m_operationGraph,
                2000,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: Check and Set CUDNN_ATTR_OPERATIONGRAPH_HANDLE");
            return std::move(m_operationGraph);
        }

        // Create a descriptor. Memory allocation happens here.
        /*
        DPCT1007:567: Migration of CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR is not supported.
        */
        auto status = m_operationGraph.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operationGraph, status, "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_operationGraph);
        }

        std::array<cudnnBackendDescriptor_t, 50> ops_raw{nullptr};
        for (auto i = 0u; i < m_operationGraph.numOps; i++) {
            ops_raw[i] = m_operationGraph.ops[i]->get_backend_descriptor();
        }

        status =
            cudnn_frontend::set_attribute(m_operationGraph.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:568: Migration of CUDNN_ATTR_OPERATIONGRAPH_OPS is not supported.
                                          */
                                          CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                          /*
                                          DPCT1007:569: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          m_operationGraph.numOps,
                                          ops_raw.data());
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operationGraph,
                status,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute CUDNN_ATTR_OPERATIONGRAPH_OPS Failed");
            return std::move(m_operationGraph);
        }
        status =
            cudnn_frontend::set_attribute(m_operationGraph.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:570: Migration of CUDNN_ATTR_OPERATIONGRAPH_HANDLE is not supported.
                                          */
                                          CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                          /*
                                          DPCT1007:571: Migration of CUDNN_TYPE_HANDLE is not supported.
                                          */
                                          CUDNN_TYPE_HANDLE,
                                          1,
                                          &m_operationGraph.handle);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operationGraph,
                status,
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: SetAttribute CUDNN_ATTR_OPERATIONGRAPH_HANDLE Failed");
            return std::move(m_operationGraph);
        }

        // Finalizing the descriptor
        status = cudnn_frontend::finalize(m_operationGraph.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(
                &m_operationGraph, status, "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_operationGraph);
        }

        getLogger() << "[cudnn_frontend] " << m_operationGraph << std::endl;
        return std::move(m_operationGraph);
    }

    explicit OperationGraphBuilder_v8()                        = default;
    ~OperationGraphBuilder_v8()                                = default;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 &&)      = delete;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 const &) = delete;
    OperationGraphBuilder_v8 &
    operator=(OperationGraphBuilder_v8 const &) = delete;

   private:
    OperationGraph_v8 m_operationGraph;
};

using OperationGraph        = OperationGraph_v8;
using OperationGraphBuilder = OperationGraphBuilder_v8;

}  // namespace cudnn_frontend
