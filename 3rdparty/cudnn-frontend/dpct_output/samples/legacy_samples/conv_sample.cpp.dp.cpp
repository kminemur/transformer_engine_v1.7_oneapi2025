#include <dpct/dnnl_utils.hpp>
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

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "conv_sample.h"
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>

namespace {

bool
isNonDeterministic(cudnnBackendDescriptor_t engine_config) {
    /*
    DPCT1007:850: Migration of CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC is not supported.
    */
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(engine_config);
}

bool
isReducedPrecisionReduction(cudnnBackendDescriptor_t engine_config) {
    /*
    DPCT1007:851: Migration of CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION is not supported.
    */
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION>(engine_config);
}

bool
isDownConvertingInputs(cudnnBackendDescriptor_t engine_config) {
    /*
    DPCT1007:852: Migration of CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS is not supported.
    */
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(engine_config);
}

bool
isNonDeterministicOrisDownConverting(cudnnBackendDescriptor_t engine_config) {
    return isNonDeterministic(engine_config) || isDownConvertingInputs(engine_config);
}

bool
allowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

bool
allowErrata(int64_t* padA) {
    return std::all_of(padA, padA + 2, [](int64_t pad) { return pad == 0; });
}

bool
isInt8Errata(dpct::library_data_t type) {
    return type == dpct::library_data_t::real_int8;
}

}  // namespace
enum {
    X_TENSOR,
    Y_TENSOR,
    W_TENSOR,
    Z_TENSOR,
    B_TENSOR,
    AFTERADD_TENSOR,
    AFTERBIAS_TENSOR,
    AFTERCONV_TENSOR,
};

using common_conv_descriptors =
    std::tuple<cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::ConvDesc>;

using common_convbias_descriptors = std::tuple<cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor>;

common_convbias_descriptors
create_conv_add_bias_act_descriptors(int64_t* x_dim,
                                     int64_t* padA,
                                     int64_t* convstrideA,
                                     int64_t* dilationA,
                                     int64_t* w_dim,
                                     int64_t* y_dim,
                                     dpct::library_data_t dataType,
                                     dpct::library_data_t computeType) {
    (void)padA;
    (void)convstrideA;
    (void)dilationA;
    int64_t b_dim[4];
    b_dim[0] = 1;
    b_dim[1] = y_dim[1];
    b_dim[2] = 1;
    b_dim[3] = 1;

    int64_t x_stride[4];
    int64_t y_stride[4];
    int64_t w_stride[4];
    int64_t b_stride[4];

    generateStrides(w_dim, w_stride, 4, dpct::dnnl::memory_format_tag::nhwc);
    generateStrides(x_dim, x_stride, 4, dpct::dnnl::memory_format_tag::nhwc);
    generateStrides(y_dim, y_stride, 4, dpct::dnnl::memory_format_tag::nhwc);
    generateStrides(b_dim, b_stride, 4, dpct::dnnl::memory_format_tag::nhwc);

    return common_convbias_descriptors(cudnn_frontend::TensorBuilder()
                                           .setDim(4, x_dim)
                                           .setStride(4, x_stride)
                                           .setId('x')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim)
                                           .setStride(4, y_stride)
                                           .setId('y')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, w_dim)
                                           .setStride(4, w_stride)
                                           .setId('w')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim)
                                           .setStride(4, y_stride)
                                           .setId('z')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, b_dim)
                                           .setStride(4, b_stride)
                                           .setId('b')
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim)
                                           .setStride(4, y_stride)
                                           .setVirtual()
                                           .setId('A')  // after add
                                           .setAlignment(4)
                                           .setDataType(computeType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim)
                                           .setStride(4, y_stride)
                                           .setVirtual()
                                           .setId('B')  // after bias
                                           .setAlignment(4)
                                           .setDataType(computeType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim)
                                           .setStride(4, y_stride)
                                           .setId('C')  // after conv
                                           .setAlignment(4)
                                           .setVirtual()
                                           .setDataType(computeType)
                                           .build());
}

common_conv_descriptors
create_common_descriptors(int64_t* x_dim,
                          int64_t* padA,
                          int64_t* convstrideA,
                          int64_t* dilationA,
                          int64_t* w_dim,
                          int64_t* y_dim,
                          dpct::library_data_t dataType,
                          int mode) {
    const int convDim = 2;

    int64_t strideA[4];
    int64_t outstrideA[4];
    int64_t filterstrideA[4];

    generateStrides(w_dim, filterstrideA, 4, dpct::dnnl::memory_format_tag::nchw);
    generateStrides(x_dim, strideA, 4, dpct::dnnl::memory_format_tag::nchw);
    generateStrides(y_dim, outstrideA, 4, dpct::dnnl::memory_format_tag::nchw);

    return common_conv_descriptors(cudnn_frontend::TensorBuilder()
                                       .setDim(4, x_dim)
                                       .setStride(4, strideA)
                                       .setId('x')
                                       .setAlignment(4)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::TensorBuilder()
                                       .setDim(4, y_dim)
                                       .setStride(4, outstrideA)
                                       .setId('y')
                                       .setAlignment(4)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::TensorBuilder()
                                       .setDim(4, w_dim)
                                       .setStride(4, filterstrideA)
                                       .setId('w')
                                       .setAlignment(4)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::ConvDescBuilder()
                                       .setComputeType(dataType)
                                       .setMathMode(mode)
                                       .setSpatialDimCount(convDim)
                                       .setSpatialStride(convDim, convstrideA)
                                       .setPrePadding(convDim, padA)
                                       .setPostPadding(convDim, padA)
                                       .setDilation(convDim, dilationA)
                                       .build());
}

cudnn_frontend::OperationGraph
create_operation_graph(common_conv_descriptors& descriptors,
                       cudnnBackendDescriptorType_t mode,
                       dpct::dnnl::engine_ext handle_) {
    float alpha = 1.0f;
    float beta  = 0.0;

    auto op = cudnn_frontend::OperationBuilder(mode)
                  .setxDesc(std::get<X_TENSOR>(descriptors))
                  .setyDesc(std::get<Y_TENSOR>(descriptors))
                  .setwDesc(std::get<W_TENSOR>(descriptors))
                  .setcDesc(std::get<3>(descriptors))
                  .setAlpha(alpha)
                  .setBeta(beta)
                  .build();

    std::cout << "Operation is " << op.describe() << std::endl;

    std::array<cudnn_frontend::Operation const*, 1> ops = {&op};

    return cudnn_frontend::OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();
}

// Method for engine config generator based on heuristics
auto heurgen_method = [](cudnn_frontend::OperationGraph& opGraph) -> cudnn_frontend::EngineConfigList {
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                          .setOperationGraph(opGraph)
                          /*
                          DPCT1007:853: Migration of CUDNN_HEUR_MODE_INSTANT is not supported.
                          */
                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                          .build();
    std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;

    auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    cudnn_frontend::EngineConfigList filtered_configs;
    cudnn_frontend::filter(engine_configs, filtered_configs, ::allowAll);
    return filtered_configs;
};

// Method for engine config generator based on fallback list
auto fallback_method = [](cudnn_frontend::OperationGraph& opGraph) -> cudnn_frontend::EngineConfigList {
    auto fallback =
        cudnn_frontend::EngineFallbackListBuilder()
            .setOperationGraph(opGraph)
            /*
            DPCT1007:854: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
            */
            .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            .build();
    auto& fallback_list = fallback.getFallbackList();

    cudnn_frontend::EngineConfigList filtered_configs;
    // We create this filter to pre-remove configs being passed to cudnnFind.
    // This is just a sample and is not necessary
    cudnn_frontend::filter(fallback_list, filtered_configs, ::isNonDeterministic);

    return filtered_configs;
};

void
run_from_heuristics(int64_t* x_dim,
                    int64_t* padA,
                    int64_t* convstrideA,
                    int64_t* dilationA,
                    int64_t* w_dim,
                    int64_t* y_dim,
                    dpct::library_data_t dataType,
                    int mode,
                    float* devPtrX,
                    float* devPtrW,
                    float* devPtrY,
                    cudnnBackendHeurMode_t heur_mode,
                    bool expect_in_cache) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    (void)heur_mode;
    static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");
    try {
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        common_conv_descriptors descriptors =
            create_common_descriptors(x_dim, padA, convstrideA, dilationA, w_dim, y_dim, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            /*
            DPCT1007:855: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
            */
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};

        const cudnn_frontend::ExecutionPlan* cached_plan;
        if (plan_cache.get_plan_from_cache(opGraph, cached_plan)) {
            std::cout << "Cached execution plan found." << cached_plan->getTag() << std::endl;
            auto workspace_size = cached_plan->getWorkspaceSize();
            std::cout << cached_plan->describe() << " requires workspace " << workspace_size << std::endl;
            void* workspace_ptr = nullptr;
            if (workspace_size > 0) {
                checkCudaErr(
                    DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
            }
            auto variantPack = cudnn_frontend::VariantPackBuilder()
                                   .setWorkspacePointer(workspace_ptr)
                                   .setDataPointers(3, data_ptrs)
                                   .setUids(3, uids)
                                   .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;
            dpct::err1 status =
                /*
                DPCT1007:856: Migration of cudnnBackendExecute is not supported.
                */
                cudnnBackendExecute(handle_, cached_plan->get_raw_desc(), variantPack.get_raw_desc());

            if (workspace_size > 0) {
                checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
            }
            cudnn_frontend::throw_if(
                [dpct::err1]() {
                                 return (status != 0);
                },
                "Plan execute error",
                status);
        } else {
            REQUIRE(false == expect_in_cache);
            std::array<cudnn_frontend::GeneratorSource const, 1> sources = {heurgen_method};
            cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

            auto workspace_size = 100 * 1024 * 1024;  // 100 MB
            void* workspace_ptr = nullptr;
            if (workspace_size > 0) {
                checkCudaErr(
                    DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
            }

            auto variantPack = cudnn_frontend::VariantPackBuilder()
                                   .setWorkspacePointer(workspace_ptr)
                                   .setDataPointers(3, data_ptrs)
                                   .setUids(3, uids)
                                   .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;

            auto plan = generator.cudnnFindPlanAndCache<
                /*
                DPCT1007:857: Migration of CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE is not supported.
                */
                cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
                handle_, opGraph, variantPack, plan_cache);

            std::cout << "Plan tag: " << plan.getTag() << " finished in " << plan.getExecutionTime() << " ms,"
                      << " workspace: " << plan.getWorkspaceSize() << " bytes" << std::endl;

            /*
            DPCT1007:858: Migration of cudnnBackendExecute is not supported.
            */
            dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

            if (workspace_size > 0) {
                checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
            }
            cudnn_frontend::throw_if(
                [dpct::err1]() {
                                 return (status != 0);
                },
                "Plan execute error",
                status);
        }
    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    /*
    DPCT1026:859: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
    */
    if (handle_);
    return;
}

void
run_from_global_index(int64_t* x_dim,
                      int64_t* padA,
                      int64_t* convstrideA,
                      int64_t* dilationA,
                      int64_t* w_dim,
                      int64_t* y_dim,
                      dpct::library_data_t dataType,
                      int mode,
                      float* devPtrX,
                      float* devPtrW,
                      float* devPtrY) {
    dpct::dnnl::engine_ext handle_;

    try {
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        common_conv_descriptors descriptors =
            create_common_descriptors(x_dim, padA, convstrideA, dilationA, w_dim, y_dim, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph = create_operation_graph(
            /*
            DPCT1007:860: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR is not supported.
            */
            descriptors,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            handle_);
        std::cout << opGraph.describe() << std::endl;

        // We have to randomly pick one engine from [0, total_engines)
        // Selecting "0" by default
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }

        if (knobs.begin() != knobs.end()) {
            std::cout << "Updated knob choice" << std::endl;
            knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
            std::cout << knobs.begin()->describe() << std::endl;
        }
        auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
        std::cout << engine_config.describe() << std::endl;
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(nullptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:861: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    /*
    DPCT1026:862: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
    */
    if (handle_);
}

dpct::err1
run_with_external_config(int64_t* x_dim,
                         int64_t* padA,
                         int64_t* convstrideA,
                         int64_t* dilationA,
                         int64_t* w_dim,
                         int64_t* y_dim,
                         dpct::library_data_t dataType,
                         int mode,
                         float* devPtrX,
                         float* devPtrW,
                         float* devPtrY) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;

    dpct::err1 status = 0;
    try {
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        common_conv_descriptors descriptors =
            create_common_descriptors(x_dim, padA, convstrideA, dilationA, w_dim, y_dim, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            /*
            DPCT1007:863: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR is not supported.
            */
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::isNonDeterministic, filtered_configs);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;

        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        cudnn_frontend::ManagedOpaqueDescriptor plan_desc = nullptr;
        int64_t workspace_size                            = 0;
        for (auto& config : filtered_configs) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle_)
                                .setEngineConfig(config, opGraph.getTag())
                                .build();
                std::cout << "Plan tag: " << plan.getTag() << std::endl;

                workspace_size = plan.getWorkspaceSize();
                std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
                plan_desc = plan.get_desc();
            } catch (cudnn_frontend::cudnnException& e) {
                status = e.getCudnnStatus();
                continue;
            }
        }
        if (plan_desc == nullptr) {
            std::cout << "No plan found implementing the operation graph" << std::endl;
            return status;
        }

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:864: Migration of cudnnBackendExecute is not supported.
        */
        status = cudnnBackendExecute(handle_, plan_desc->get_backend_descriptor(), variantPack.get_raw_desc());
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << " " << cudnn_frontend::to_string(e.getCudnnStatus())
                  << std::endl;
        CHECK(false);
    }

    /*
    DPCT1026:865: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
    */
    if (handle_);

    return status;
}

// create_plan(std::vector<cudnnBackendDescriptor_t> &)
void
run_conv_add_bias_activation(int64_t* x_dim,
                             int64_t* pad,
                             int64_t* convstride,
                             int64_t* dilation,
                             int64_t* w_dim,
                             int64_t* y_dim,
                             dpct::library_data_t dataType,
                             float* devPtrX,
                             float* devPtrW,
                             float* devPtrY,
                             float* devPtrZ,
                             float* devPtrB) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        int convDim = 2;
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        common_convbias_descriptors tensors = create_conv_add_bias_act_descriptors(
            x_dim, pad, convstride, dilation, w_dim, y_dim, dataType, dpct::library_data_t::real_float);
        std::cout << std::get<X_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<Z_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<B_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERADD_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERBIAS_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERCONV_TENSOR>(tensors).describe() << std::endl;

        // Define the add operation
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:866: Migration of CUDNN_POINTWISE_ADD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(dpct::library_data_t::real_float)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // Define the bias operation
        auto addDesc2 = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:867: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(dpct::library_data_t::real_float)
                            .build();
        std::cout << addDesc2.describe() << std::endl;

        // Define the activation operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:868: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(0)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha  = 1.0f;
        float alpha2 = 0.5f;
        float beta   = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:869: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(std::get<X_TENSOR>(tensors))
                           .setwDesc(std::get<W_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Add Node with scaling parameters.
        /*
        DPCT1007:870: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto add_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(std::get<Z_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERADD_TENSOR>(tensors))
                           .setpwDesc(addDesc)
                           .setAlpha(alpha)
                           .setAlpha2(alpha2)
                           .build();
        std::cout << add_op1.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:871: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto add_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op1.getOutputTensor())
                           .setbDesc(std::get<B_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERBIAS_TENSOR>(tensors))
                           .setpwDesc(addDesc2)
                           .build();
        std::cout << add_op2.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:872: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_op2.getOutputTensor())
                          .setyDesc(std::get<Y_TENSOR>(tensors))
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &add_op1, &add_op2, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // How many engines support this operation graph ?
        auto total_engines = opGraph.getEngineCount();
        std::cout << "conv_add_bias_activation " << opGraph.describe() << " has " << total_engines << " engines."
                  << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::isNonDeterministic, filtered_configs);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        for (auto& filtered_config : filtered_configs) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle_)
                                .setEngineConfig(filtered_config, opGraph.getTag())
                                .build();
                std::cout << "Plan tag: " << plan.getTag() << std::endl;

                auto workspace_size = plan.getWorkspaceSize();
                std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

                void* workspace_ptr = nullptr;
                if (workspace_size > 0) {
                    checkCudaErr(
                        DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
                }
                void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB};
                int64_t uids[]    = {'x', 'y', 'w', 'z', 'b'};
                auto variantPack  = cudnn_frontend::VariantPackBuilder()
                                       .setWorkspacePointer(workspace_ptr)
                                       .setDataPointers(5, data_ptrs)
                                       .setUids(5, uids)
                                       .build();
                std::cout << "variantPack " << variantPack.describe() << std::endl;
                /*
                DPCT1007:873: Migration of cudnnBackendExecute is not supported.
                */
                dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
                if (workspace_size > 0) {
                    checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
                }
                cudnn_frontend::throw_if(
                    [dpct::err1]() {
                                     return (status != 0);
                    },
                    "Plan execute error",
                    status);
                std::cout << "Test completed succesfully" << std::endl;
                return;
            } catch (cudnn_frontend::cudnnException& e) {
                continue;
            }
        }

    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_from_cudnn_find(int64_t* x_dim,
                    int64_t* padA,
                    int64_t* convstrideA,
                    int64_t* dilationA,
                    int64_t* w_dim,
                    int64_t* y_dim,
                    dpct::library_data_t dataType,
                    int mode,
                    void* devPtrX,
                    void* devPtrW,
                    void* devPtrY) {
    dpct::dnnl::engine_ext handle_;

    try {
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        common_conv_descriptors descriptors =
            create_common_descriptors(x_dim, padA, convstrideA, dilationA, w_dim, y_dim, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            /*
            DPCT1007:874: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
            */
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder().setDataPointers(3, data_ptrs).setUids(3, uids).build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        auto sample_predicate_function = [](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            const int32_t max_plan_count = 5;
            static int32_t plan_count    = 0;

            // Filter out plans that require non-zero workspace
            if (plan.getWorkspaceSize() != 0) return true;

            plan_count++;

            // Only keep first 5 plans
            return plan_count > max_plan_count;
        };

        std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
        cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

        auto options =
            /*
            DPCT1007:875: Migration of CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE is not supported.
            */
            generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
                handle_, opGraph, variantPack, sample_predicate_function);

        std::for_each(options.begin(), options.end(), [](cudnn_frontend::ExecutionPlan& opt) {
            std::cout << "Plan tag: " << opt.getTag() << " finished in " << opt.getExecutionTime() << " ms,"
                      << " workspace: " << opt.getWorkspaceSize() << " bytes" << std::endl;
        });

        /*
        DPCT1007:876: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, options.front().get_raw_desc(), variantPack.get_raw_desc());

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    /*
    DPCT1026:877: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
    */
    if (handle_);
    return;
}

void
run_conv_add_bias_activation_with_cudnn_find(int64_t* x_dim,
                                             int64_t* pad,
                                             int64_t* convstride,
                                             int64_t* dilation,
                                             int64_t* w_dim,
                                             int64_t* y_dim,
                                             dpct::library_data_t dataType,
                                             float* devPtrX,
                                             float* devPtrW,
                                             float* devPtrY,
                                             float* devPtrZ,
                                             float* devPtrB) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        int convDim = 2;
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        common_convbias_descriptors tensors = create_conv_add_bias_act_descriptors(
            x_dim, pad, convstride, dilation, w_dim, y_dim, dataType, dpct::library_data_t::real_float);
        std::cout << std::get<X_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<Z_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<B_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERADD_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERBIAS_TENSOR>(tensors).describe() << std::endl;
        std::cout << std::get<AFTERCONV_TENSOR>(tensors).describe() << std::endl;

        // Define the add operation
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:878: Migration of CUDNN_POINTWISE_ADD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(dpct::library_data_t::real_float)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // Define the bias operation
        auto addDesc2 = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:879: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(dpct::library_data_t::real_float)
                            .build();
        std::cout << addDesc2.describe() << std::endl;

        // Define the activation operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:880: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(0)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha  = 1.0f;
        float alpha2 = 0.5f;
        float beta   = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:881: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(std::get<X_TENSOR>(tensors))
                           .setwDesc(std::get<W_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Add Node with scaling parameters.
        /*
        DPCT1007:882: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto add_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(std::get<Z_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERADD_TENSOR>(tensors))
                           .setpwDesc(addDesc)
                           .setAlpha(alpha)
                           .setAlpha2(alpha2)
                           .build();
        std::cout << add_op1.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:883: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto add_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op1.getOutputTensor())
                           .setbDesc(std::get<B_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERBIAS_TENSOR>(tensors))
                           .setpwDesc(addDesc2)
                           .build();
        std::cout << add_op2.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:884: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_op2.getOutputTensor())
                          .setyDesc(std::get<Y_TENSOR>(tensors))
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &add_op1, &add_op2, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto max_workspace_size = 10 * 1024 * 1024;  // 10 MiB
        void* workspace_ptr     = nullptr;
        checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device(max_workspace_size, q_ct1)));

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB};
        int64_t uids[]    = {'x', 'y', 'w', 'z', 'b'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(5, data_ptrs)
                               .setUids(5, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        auto sample_predicate_function = [=](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            return plan.getWorkspaceSize() > max_workspace_size;
        };

        std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
        cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

        auto options =
            /*
            DPCT1007:885: Migration of CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE is not supported.
            */
            generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
                handle_, opGraph, variantPack, sample_predicate_function);

        std::for_each(options.begin(), options.end(), [](cudnn_frontend::ExecutionPlan& opt) {
            std::cout << "Plan tag: " << opt.getTag() << " finished in " << opt.getExecutionTime() << " ms,"
                      << " workspace: " << opt.getWorkspaceSize() << " bytes" << std::endl;
        });

        /*
        DPCT1007:886: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, options.front().get_raw_desc(), variantPack.get_raw_desc());

        checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_from_cudnn_get(int64_t* x_dim,
                   int64_t* padA,
                   int64_t* convstrideA,
                   int64_t* dilationA,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   dpct::library_data_t dataType,
                   int mode,
                   float* devPtrX,
                   float* devPtrW,
                   float* devPtrY) {
    dpct::dnnl::engine_ext handle_;

    try {
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        common_conv_descriptors descriptors =
            create_common_descriptors(x_dim, padA, convstrideA, dilationA, w_dim, y_dim, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            /*
            DPCT1007:887: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
            */
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;

        auto sample_predicate_function = [](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            (void)plan;
            return false;
        };

        std::array<cudnn_frontend::GeneratorSource const, 1> sources = {heurgen_method};
        cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

        auto plans = generator.cudnnGetPlan(handle_, opGraph, sample_predicate_function);

        int64_t max_workspace_size = 0u;
        std::for_each(plans.begin(), plans.end(), [&max_workspace_size](cudnn_frontend::ExecutionPlan& plan) {
            std::cout << "Plan tag: " << plan.getTag() << " workspace: " << plan.getWorkspaceSize() << " bytes"
                      << std::endl;
            if (plan.getWorkspaceSize() > max_workspace_size) {
                max_workspace_size = plan.getWorkspaceSize();
            }
        });

        std::cout << "Max workspace size required " << max_workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        checkCudaErr(DPCT_CHECK_ERROR(
            workspace_ptr = (void*)sycl::malloc_device((size_t)max_workspace_size, dpct::get_in_order_queue())));

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // This is an optional step in this test.
        // time_sorted_plan makes this equivalent to using find for autotuning, and this step is not necessary if the
        // intent is to just use the heuristics.
        auto options = cudnn_frontend::time_sorted_plan<
            /*
            DPCT1007:888: Migration of CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE is not supported.
            */
            cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            handle_, std::move(plans), variantPack);

        std::for_each(options.begin(), options.end(), [](cudnn_frontend::ExecutionPlan& opt) {
            std::cout << "Plan tag: " << opt.getTag() << " finished in " << opt.getExecutionTime() << " ms,"
                      << " workspace: " << opt.getWorkspaceSize() << " bytes" << std::endl;
        });

        /*
        DPCT1007:889: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, options.front().get_raw_desc(), variantPack.get_raw_desc());

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    /*
    DPCT1026:890: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
    */
    if (handle_);
    return;
}

void
block_using_errata(int64_t* x_dim,
                   int64_t* padA,
                   int64_t* convstrideA,
                   int64_t* dilationA,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   dpct::library_data_t dataType,
                   int mode,
                   float* devPtrX,
                   float* devPtrW,
                   float* devPtrY) {
    dpct::dnnl::engine_ext handle_;

    try {
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        common_conv_descriptors descriptors =
            create_common_descriptors(x_dim, padA, convstrideA, dilationA, w_dim, y_dim, dataType, mode);

        (void)devPtrX;
        (void)devPtrY;
        (void)devPtrW;

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph = create_operation_graph(
            /*
            DPCT1007:891: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR is not supported.
            */
            descriptors,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
            handle_);
        std::cout << opGraph.describe() << std::endl;

        // We have to randomly pick one engine from [0, total_engines)
        // Selecting "0" by default
        auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(20).setOperationGraph(opGraph).build();
        std::cout << engine.describe() << std::endl;
        auto& knobs = engine.getSupportedKnobs();
        for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
            std::cout << it->describe() << std::endl;
        }

        if (knobs.begin() != knobs.end()) {
            std::cout << "Updated knob choice" << std::endl;
            knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
            std::cout << knobs.begin()->describe() << std::endl;
        }
        auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
        std::cout << engine_config.describe() << std::endl;
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        /// Please note that the json string mentioned below is just an example and is
        /// not actually a buggy engine config (kernel).
        auto json_handle = json::parse(R"(
            { "version" : 1, 
              "rules"   : 
                [ 
                    { "rule_id"             : "ConvBwdData_eng1_k2=2_k3=0", 
                      "operation"           : "ConvBwdData",
                      "engine"              : 1, 
                      "knob"                : ["k2=4", "k3=0"],
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : -1 
                    }, 
                    { "rule_id"             : "ConvBwdFilter_eng20",
                      "operation"           : "ConvBwdFilter",
                      "engine"              : 20,
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : -1 
                    } 
                ] 
            })");

        auto fn              = std::bind(::allowErrata, padA);
        bool is_plan_blocked = cudnn_frontend::check_errata<decltype(fn)>(json_handle, plan.getTag(), handle_, fn);
        CHECK(is_plan_blocked);

        // Filter kernels with specific shape
        auto json_handle_with_shape = json::parse(R"(
            { "version" : 1, 
              "rules"   : 
                [ 
                    { "rule_id"             : "ConvBwdData_eng1_k2=2_k3=0", 
                      "operation"           : "ConvBwdData",
                      "engine"              : 1, 
                      "knob"                : ["k2=4", "k3=0"],
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : -1 
                    }, 
                    { "rule_id"             : "ConvBwdFilter_eng20",
                      "operation"           : "ConvBwdFilter",
                      "engine"              : 20,
                      "shape_format"        : "NCHW",
                      "input_shape"         : [1, 32, 128, 128],
                      "filter_shape"        : [32, 32, 3, 3],
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : -1 
                    } 
                ] 
            })");

        is_plan_blocked =
            cudnn_frontend::check_errata<decltype(fn)>(json_handle_with_shape, plan.getTag(), handle_, opGraph, fn);
        CHECK(is_plan_blocked);

        // Filter kernels only on spatial dims (wildcard usage)
        auto json_handle_with_shape_wildcards = json::parse(R"(
            { "version" : 1, 
              "rules"   : 
                [ 
                    { "rule_id"             : "ConvBwdData_eng1_k2=2_k3=0", 
                      "operation"           : "ConvBwdData",
                      "engine"              : 1, 
                      "knob"                : ["k2=4", "k3=0"],
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : -1 
                    }, 
                    { "rule_id"             : "ConvBwdFilter_eng20",
                      "operation"           : "ConvBwdFilter",
                      "engine"              : 20,
                      "shape_format"        : "NCHW",
                      "input_shape"         : [-1, -1, 128, 128],
                      "filter_shape"        : [-1, -1, 3, 3],
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : -1 
                    } 
                ] 
            })");

        is_plan_blocked = cudnn_frontend::check_errata<decltype(fn)>(
            json_handle_with_shape_wildcards, plan.getTag(), handle_, opGraph, fn);
        CHECK(is_plan_blocked);

    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }

    /*
    DPCT1026:892: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
    */
    if (handle_);
}

void
run_dp4a(int64_t* x_dim,
         int64_t* padA,
         int64_t* convstrideA,
         int64_t* dilationA,
         int64_t* w_dim,
         int64_t* y_dim,
         int mode,
         void* devPtrX,
         void* devPtrW,
         void* devPtrY,
         int64_t vectorCount,
         int64_t vectorDimension) {
    dpct::dnnl::engine_ext handle_;

    try {
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        const int convDim = 2;
        (void)convDim;

        int64_t strideA[4];
        int64_t outstrideA[4];
        int64_t filterstrideA[4];

        generateStrides(w_dim, filterstrideA, 4, dpct::dnnl::memory_format_tag::nchw);
        generateStrides(x_dim, strideA, 4, dpct::dnnl::memory_format_tag::nchw);
        generateStrides(y_dim, outstrideA, 4, dpct::dnnl::memory_format_tag::nchw);

        auto tensor_x = cudnn_frontend::TensorBuilder()
                            .setDim(4, x_dim)
                            .setStride(4, strideA)
                            .setId('x')
                            .setAlignment(16)
                            .setDataType(dpct::library_data_t::real_int8)
                            .setVectorCountAndDimension(vectorCount, vectorDimension)
                            .build();
        auto tensor_y = cudnn_frontend::TensorBuilder()
                            .setDim(4, y_dim)
                            .setStride(4, outstrideA)
                            .setId('y')
                            .setAlignment(16)
                            .setDataType(dpct::library_data_t::real_int8)
                            .setVectorCountAndDimension(vectorCount, vectorDimension)
                            .build();
        auto tensor_w = cudnn_frontend::TensorBuilder()
                            .setDim(4, w_dim)
                            .setStride(4, filterstrideA)
                            .setId('w')
                            .setAlignment(16)
                            .setDataType(dpct::library_data_t::real_int8)
                            .setVectorCountAndDimension(vectorCount, vectorDimension)
                            .build();
        auto conv_desc = cudnn_frontend::ConvDescBuilder()
                             .setComputeType(dpct::library_data_t::real_int32)
                             .setMathMode(mode)
                             .setSpatialDimCount(convDim)
                             .setSpatialStride(convDim, convstrideA)
                             .setPrePadding(convDim, padA)
                             .setPostPadding(convDim, padA)
                             .setDilation(convDim, dilationA)
                             .build();
        float alpha = 1.0f;
        float beta  = 0.0f;
        /*
        DPCT1007:893: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                      .setxDesc(tensor_x)
                      .setyDesc(tensor_y)
                      .setwDesc(tensor_w)
                      .setcDesc(conv_desc)
                      .setAlpha(alpha)
                      .setBeta(beta)
                      .build();
        std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        auto max_workspace_size = 1024 * 1024 * 1024;  // 1 GB
        void* workspace_ptr     = nullptr;
        checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr =
                                          (void*)sycl::malloc_device(max_workspace_size, dpct::get_in_order_queue())));

        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .setWorkspacePointer(workspace_ptr)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        auto sample_predicate_function = [max_workspace_size](cudnn_frontend::ExecutionPlan const& plan) -> bool {
            return plan.getWorkspaceSize() > max_workspace_size;
        };

        std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
        cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

        auto options =
            /*
            DPCT1007:894: Migration of CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE is not supported.
            */
            generator.cudnnFindPlan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
                handle_, opGraph, variantPack, sample_predicate_function);

        std::for_each(options.begin(), options.end(), [](cudnn_frontend::ExecutionPlan& opt) {
            std::cout << "Plan tag: " << opt.getTag() << " finished in " << opt.getExecutionTime() << " ms,"
                      << " workspace: " << opt.getWorkspaceSize() << " bytes" << std::endl;
        });

        /*
        DPCT1007:895: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, options.front().get_raw_desc(), variantPack.get_raw_desc());

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
    /*
    DPCT1026:896: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
    */
    if (handle_);
}

void
run_imma(int64_t* x_dim_padded,
         int64_t* padA,
         int64_t* convstrideA,
         int64_t* dilationA,
         int64_t* w_dim_padded,
         int64_t* y_dim_padded,
         int mode,
         void* devPtrX,
         void* devPtrW,
         void* devPtrY,
         int64_t vectorCount,
         int64_t vectorDimension) try {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;

    try {
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        const int convDim = 2;
        (void)convDim;

        int64_t strideA_padded[4];
        int64_t outstrideA_padded[4];
        int64_t filterstrideA_padded[4];

        generateStrides(w_dim_padded, filterstrideA_padded, 4, dpct::dnnl::memory_format_tag::nchw);
        generateStrides(x_dim_padded, strideA_padded, 4, dpct::dnnl::memory_format_tag::nchw);
        generateStrides(y_dim_padded, outstrideA_padded, 4, dpct::dnnl::memory_format_tag::nchw);

        auto tensor_x = cudnn_frontend::TensorBuilder()
                            .setDim(4, x_dim_padded)
                            .setStride(4, strideA_padded)
                            .setId('x')
                            .setAlignment(16)
                            .setDataType(dpct::library_data_t::real_int8)
                            .setVectorCountAndDimension(vectorCount, vectorDimension)
                            .build();
        auto tensor_y = cudnn_frontend::TensorBuilder()
                            .setDim(4, y_dim_padded)
                            .setStride(4, outstrideA_padded)
                            .setId('y')
                            .setAlignment(16)
                            .setDataType(dpct::library_data_t::real_int8)
                            .setVectorCountAndDimension(vectorCount, vectorDimension)
                            .build();
        auto tensor_w = cudnn_frontend::TensorBuilder()
                            .setDim(4, w_dim_padded)
                            .setStride(4, filterstrideA_padded)
                            .setId('w')
                            .setAlignment(16)
                            .setDataType(dpct::library_data_t::real_int8)
                            .setReorderType(cudnn_frontend::TensorReordering_t::INT8x32)
                            .setVectorCountAndDimension(vectorCount, vectorDimension)
                            .build();
        auto conv_desc = cudnn_frontend::ConvDescBuilder()
                             .setComputeType(dpct::library_data_t::real_int32)
                             .setMathMode(mode)
                             .setSpatialDimCount(convDim)
                             .setSpatialStride(convDim, convstrideA)
                             .setPrePadding(convDim, padA)
                             .setPostPadding(convDim, padA)
                             .setDilation(convDim, dilationA)
                             .build();
        std::cout << tensor_x.describe() << std::endl;
        std::cout << tensor_w.describe() << std::endl;
        std::cout << tensor_y.describe() << std::endl;
        float alpha = 1.0f;
        float beta  = 0.0f;
        /*
        DPCT1007:897: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                      .setxDesc(tensor_x)
                      .setyDesc(tensor_y)
                      .setwDesc(tensor_w)
                      .setcDesc(conv_desc)
                      .setAlpha(alpha)
                      .setBeta(beta)
                      .build();
        std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        auto max_workspace_size = 1024 * 1024 * 1024;  // 1 GB
        void* workspace_ptr     = nullptr;
        checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device(max_workspace_size, q_ct1)));

        auto engine_configs_h = heurgen_method(opGraph);
        auto engine_configs_f = fallback_method(opGraph);

        cudnn_frontend::EngineConfigList filtered_configs;
        cudnn_frontend::filter(engine_configs_h, filtered_configs, ::allowAll);
        cudnn_frontend::filter(engine_configs_f, filtered_configs, ::allowAll);
        std::cout << "filtered_configs " << filtered_configs.size() << std::endl;

        cudnn_frontend::executionPlans_t options;
        for (auto& cfg : filtered_configs) {
            try {
                options.emplace_back(cudnn_frontend::ExecutionPlanBuilder()
                                         .setHandle(handle_)
                                         .setEngineConfig(cfg, opGraph.getTag())
                                         .build());
            } catch (cudnn_frontend::cudnnException&) {
                continue;
            }
        }

        std::for_each(options.begin(), options.end(), [](cudnn_frontend::ExecutionPlan& opt) {
            std::cout << "Plan tag: " << opt.getTag() << " finished in " << opt.getExecutionTime() << " ms,"
                      << " workspace: " << opt.getWorkspaceSize() << " bytes." << std::endl;
        });

        int64_t filter_size = tensor_w.getPackedElementCount();
        void* reorderedData = nullptr;

        auto cuda_status = DPCT_CHECK_ERROR(reorderedData = (void*)sycl::malloc_device((size_t)filter_size, q_ct1));
        CHECK(cuda_status == 0);

        auto reorder_status = cudnn_frontend::cudnnReorderFilterAndBiasInt8x32(
            handle_, tensor_w, conv_desc, devPtrW, reorderedData, nullptr, nullptr);
        CHECK(reorder_status == 0);

        void* data_ptrs[] = {devPtrX, devPtrY, reorderedData};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .setWorkspacePointer(workspace_ptr)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        if (options.size() == 0) {
            return;
        }

        auto json_handle = json::parse(R"(
            { "version" : 1, 
              "rules"   : 
                [ 
                    { "rule_id"             : "ConvFwd_eng0", 
                      "operation"           : "ConvFwd",
                      "engine"              : 0, 
                      "knob"                : [],
                      "cudnn_version_start" : 8000, 
                      "cudnn_version_end"   : 8300 
                    }
                ] 
            })");

        auto fn = std::bind(::isInt8Errata, dpct::library_data_t::real_int8);

        dpct::err1 status = 0;

        for (auto& option : options) {
            bool is_plan_blocked =
                cudnn_frontend::check_errata<decltype(fn)>(json_handle, option.getTag(), handle_, fn);
            if (is_plan_blocked) {
                continue;
            }

            std::cout << "Executing " << option.getTag() << std::endl;
            /*
            DPCT1007:898: Migration of cudnnBackendExecute is not supported.
            */
            status = cudnnBackendExecute(handle_, option.get_raw_desc(), variantPack.get_raw_desc());
        }

        dpct::dpct_free(reorderedData, q_ct1);
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
    } catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
    /*
    DPCT1026:899: The call to cudnnDestroy was removed because this functionality is redundant in SYCL.
    */
    if (handle_);
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
