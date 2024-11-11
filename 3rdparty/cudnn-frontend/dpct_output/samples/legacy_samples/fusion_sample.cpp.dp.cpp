#include <dpct/dnnl_utils.hpp>
/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "fusion_sample.h"
#include <cudnn_frontend.h>
#include "../utils/error_util.h"

bool
allowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

#if (CUDNN_VERSION >= 8200)
bool
isRuntimeCompilation(cudnnBackendDescriptor_t engine_config) {
    /*
    DPCT1007:990: Migration of CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION is not supported.
    */
    return cudnn_frontend::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION>(engine_config);
}
#endif

cudnn_frontend::ExecutionPlan
get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, dpct::dnnl::engine_ext handle_) {
#if (CUDNN_VERSION >= 8200)
    {
        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                              .setOperationGraph(opGraph)
                              /*
                              DPCT1007:991: Migration of CUDNN_HEUR_MODE_INSTANT is not supported.
                              */
                              .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                              .build();

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

        // Try engine configs returned by the heuristics and pick up the first one that works.
        for (auto& ecfg : engine_config) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle_)
                                .setEngineConfig(ecfg, opGraph.getTag())
                                .build();
                return plan;
            } catch (cudnn_frontend::cudnnException& e) {
                continue;
            }
        }
    }
#endif

    {
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<1>(
            {"heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (auto status : statuses) {
            std::cout << cudnn_frontend::to_string(status) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        return cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle_)
            .setEngineConfig(filtered_configs[0], opGraph.getTag())
            .build();
    }
}

void
run_conv_scale_bias_add_leaky_relu(int64_t* x_dim,
                                   int64_t* w_dim,
                                   int64_t* y_dim,
                                   int64_t* s_dim,
                                   int64_t* b_dim,
                                   int64_t* a_dim,
                                   dpct::library_data_t dataType,
                                   int convDim,
                                   int64_t* conv_padA,
                                   int64_t* conv_dilationA,
                                   int64_t* conv_strideA,
                                   void* devPtrX,
                                   void* devPtrW,
                                   void* devPtrY,
                                   void* devPtrS,
                                   void* devPtrB,
                                   void* devPtrA) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(s_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(b_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(a_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto aTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, a_dim)
                           .setStride(4, stride)
                           .setId('a')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterAddTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, y_dim)
                                  .setStride(4, stride)
                                  .setId('D')  // after add
                                  .setAlignment(16)
                                  .setVirtual()
                                  .setDataType(dataType)
                                  .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << aTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterAddTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             /*
                             DPCT1007:992: Migration of CUDNN_POINTWISE_MUL is not supported.
                             */
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:993: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(dpct::library_data_t::real_float)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the add descriptor
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:994: Migration of CUDNN_POINTWISE_ADD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:995: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .setReluLowerClipSlope(0.01)  // leaky relu
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:996: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:997: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(conv_op.getOutputTensor())
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:998: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(scale_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create a Add Node.
        /*
        DPCT1007:999: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setbDesc(aTensor)
                          .setyDesc(afterAddTensor)
                          .setpwDesc(addDesc)
                          .build();
        std::cout << add_op.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:1000: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution scale bias add activation
        std::array<cudnn_frontend::Operation const*, 5> ops = {&conv_op, &scale_op, &bias_op, &add_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrS, devPtrB, devPtrA};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'b', 'a'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(6, data_ptrs)
                               .setUids(6, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1001: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1002: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere cards
        /*
        DPCT1005:1003: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && e.getCudnnStatus() == 3000) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
        } else {
#if (CUDNN_VERSION == 8600) || (CUDNN_VERSION == 8700)
            if (prop.major == 9) {
                std::cout << "Hopper GPUs does not have float fused operations support yet\n";
                return;
            }
#endif
#if (CUDNN_VERSION >= 8300)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif
        }
    }
}

void
run_conv_bias_scale_relu(int64_t* x_dim,
                         int64_t* w_dim,
                         int64_t* y_dim,
                         int64_t* b_dim,
                         int64_t* s_dim,
                         dpct::library_data_t dataType,
                         int convDim,
                         int64_t* conv_padA,
                         int64_t* conv_dilationA,
                         int64_t* conv_strideA,
                         void* devPtrX,
                         void* devPtrW,
                         void* devPtrY,
                         void* devPtrB,
                         void* devPtrS) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('C')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:1004: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(dpct::library_data_t::real_float)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             /*
                             DPCT1007:1005: Migration of CUDNN_POINTWISE_MUL is not supported.
                             */
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1006: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1007: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:1008: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1009: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(bias_op.getOutputTensor())
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:1010: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(scale_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &bias_op, &scale_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrB, devPtrS};
        int64_t uids[]    = {'x', 'y', 'w', 'b', 's'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(5, data_ptrs)
                               .setUids(5, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1011: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1012: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        // this example is only for Ampere cards
        /*
        DPCT1005:1013: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
            CHECK(false);
#endif
        }
    }
}

void
run_serialization_conv_bias_scale_relu(int64_t* x_dim,
                                       int64_t* w_dim,
                                       int64_t* y_dim,
                                       int64_t* b_dim,
                                       int64_t* s_dim,
                                       dpct::library_data_t dataType,
                                       int convDim,
                                       int64_t* conv_padA,
                                       int64_t* conv_dilationA,
                                       int64_t* conv_strideA,
                                       void* devPtrX,
                                       void* devPtrW,
                                       void* devPtrY,
                                       void* devPtrB,
                                       void* devPtrS) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('C')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dataType)
                                    .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:1014: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(dpct::library_data_t::real_float)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             /*
                             DPCT1007:1015: Migration of CUDNN_POINTWISE_MUL is not supported.
                             */
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1016: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1017: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:1018: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1019: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(bias_op.getOutputTensor())
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:1020: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(scale_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &bias_op, &scale_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        std::string plan_json;
        {
            // Suppose this is how execution plans are normally created
            auto plan_tmp = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);
            // Generate a JSON serialization of the execution plan
            plan_json = plan_tmp.getJsonRepresentation();
            // Optionally save to a file, etc...
            // std::ofstream output_file("execution_plan.json");
            // output_file << plan_json;
            // The temporary execution plan can now be discarded.
        }
        // Load the plan from a JSON string.
        auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).loadFromJson(plan_json);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrB, devPtrS};
        int64_t uids[]    = {'x', 'y', 'w', 'b', 's'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(5, data_ptrs)
                               .setUids(5, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1021: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1022: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        // this example is only for Ampere cards
        /*
        DPCT1005:1023: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
#if (CUDNN_VERSION >= 8400)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif
        }
    }
}

void
run_conv_scale_bias_relu_gen_index_selection(int64_t* x_dim,
                                             int64_t* w_dim,
                                             int64_t* y_dim,
                                             int64_t* s_dim,
                                             int64_t* b_dim,
                                             int64_t* threshold_dim,
                                             dpct::library_data_t dataType,
                                             int convDim,
                                             int64_t* conv_padA,
                                             int64_t* conv_dilationA,
                                             int64_t* conv_strideA,
                                             int axis,
                                             void* devPtrX,
                                             void* devPtrW,
                                             void* devPtrY,
                                             void* devPtrS,
                                             void* devPtrB,
                                             void* devPtrTopThreshold,
                                             void* devPtrBottomThreshold) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    (void)handle_;
    (void)x_dim;
    (void)w_dim;
    (void)y_dim;
    (void)s_dim;
    (void)b_dim;
    (void)threshold_dim;
    (void)dataType;
    (void)convDim;
    (void)conv_padA;
    (void)conv_dilationA;
    (void)conv_strideA;
    (void)axis;
    (void)devPtrX;
    (void)devPtrW;
    (void)devPtrY;
    (void)devPtrS;
    (void)devPtrB;
    (void)devPtrTopThreshold;
    (void)devPtrBottomThreshold;
    try {
#if (CUDNN_VERSION >= 8400)
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("turing") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_conv_scale_bias_relu_gen_index_selection: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();
        generateStrides(s_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(b_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dpct::library_data_t::real_float)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dpct::library_data_t::real_float)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dpct::library_data_t::real_float)
                                   .build();

        auto afterActivationTensor = cudnn_frontend::TensorBuilder()
                                         .setDim(4, y_dim)
                                         .setStride(4, stride)
                                         .setId('D')  // after activation
                                         .setAlignment(16)
                                         .setVirtual()
                                         .setDataType(dpct::library_data_t::real_float)
                                         .build();

        auto genIndexTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, y_dim)
                                  .setStride(4, stride)
                                  .setId('I')  // output of the gen index operation
                                  .setAlignment(16)
                                  .setVirtual()
                                  .setDataType(dpct::library_data_t::real_int32)
                                  .build();

        auto maskTopTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(4, y_dim)
                                 .setStride(4, stride)
                                 .setId('m')  // top half of the mask created after the less than
                                 .setAlignment(16)
                                 .setVirtual()
                                 /*
                                 DPCT1007:1024: Migration of CUDNN_DATA_BOOLEAN is not supported.
                                 */
                                 .setDataType(CUDNN_DATA_BOOLEAN)
                                 .build();

        auto maskBottomTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('n')  // bottom half of the mask
                                    .setAlignment(16)
                                    .setVirtual()
                                    /*
                                    DPCT1007:1025: Migration of CUDNN_DATA_BOOLEAN is not supported.
                                    */
                                    .setDataType(CUDNN_DATA_BOOLEAN)
                                    .build();

        auto maskTensor = cudnn_frontend::TensorBuilder()
                              .setDim(4, y_dim)
                              .setStride(4, stride)
                              .setId('M')  // OR of the top and bottom masks
                              .setAlignment(16)
                              .setVirtual()
                              /*
                              DPCT1007:1026: Migration of CUDNN_DATA_BOOLEAN is not supported.
                              */
                              .setDataType(CUDNN_DATA_BOOLEAN)
                              .build();

        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(threshold_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto thresholdTopTensor = cudnn_frontend::TensorBuilder()
                                      .setDim(4, threshold_dim)
                                      .setStride(4, stride)
                                      .setId('t')  // threshold for creating the top mask
                                      .setAlignment(16)
                                      .setDataType(dpct::library_data_t::real_int32)
                                      .build();

        auto thresholdBottomTensor = cudnn_frontend::TensorBuilder()
                                         .setDim(4, threshold_dim)
                                         .setStride(4, stride)
                                         .setId('u')  // threshold for creating the bottom mask
                                         .setAlignment(16)
                                         .setDataType(dpct::library_data_t::real_int32)
                                         .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << afterActivationTensor.describe() << std::endl;
        std::cout << genIndexTensor.describe() << std::endl;
        std::cout << maskTopTensor.describe() << std::endl;
        std::cout << maskBottomTensor.describe() << std::endl;
        std::cout << maskTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;
        std::cout << thresholdTopTensor.describe() << std::endl;
        std::cout << thresholdBottomTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             /*
                             DPCT1007:1027: Migration of CUDNN_POINTWISE_MUL is not supported.
                             */
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:1028: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(dpct::library_data_t::real_float)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1029: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the genIndex descriptor
        auto genIndexDesc = cudnn_frontend::PointWiseDescBuilder()
                                /*
                                DPCT1007:1030: Migration of CUDNN_POINTWISE_GEN_INDEX is not supported.
                                */
                                .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                .setComputeType(dpct::library_data_t::real_float)
                                .setAxis(axis)
                                .build();
        std::cout << genIndexDesc.describe() << std::endl;

        // Define the lessThan descriptor
        auto lessThanDesc = cudnn_frontend::PointWiseDescBuilder()
                                /*
                                DPCT1007:1031: Migration of CUDNN_POINTWISE_CMP_LT is not supported.
                                */
                                .setMode(CUDNN_POINTWISE_CMP_LT)
                                .setComputeType(dpct::library_data_t::real_float)
                                .build();
        std::cout << lessThanDesc.describe() << std::endl;

        // Define the greaterThan descriptor
        auto greaterThanDesc = cudnn_frontend::PointWiseDescBuilder()
                                   /*
                                   DPCT1007:1032: Migration of CUDNN_POINTWISE_CMP_GT is not supported.
                                   */
                                   .setMode(CUDNN_POINTWISE_CMP_GT)
                                   .setComputeType(dpct::library_data_t::real_float)
                                   .build();
        std::cout << greaterThanDesc.describe() << std::endl;

        // Define the logical_or descriptor
        auto logicalOrDesc = cudnn_frontend::PointWiseDescBuilder()
                                 /*
                                 DPCT1007:1034: Migration of CUDNN_POINTWISE_LOGICAL_OR is not supported.
                                 */
                                 .setMode(CUDNN_POINTWISE_LOGICAL_OR)
                                 /*
                                 DPCT1007:1033: Migration of CUDNN_DATA_BOOLEAN is not supported.
                                 */
                                 .setComputeType(CUDNN_DATA_BOOLEAN)
                                 .build();
        std::cout << logicalOrDesc.describe() << std::endl;

        // Define the binary_selection descriptor
        auto selectionDesc = cudnn_frontend::PointWiseDescBuilder()
                                 /*
                                 DPCT1007:1035: Migration of CUDNN_POINTWISE_BINARY_SELECT is not supported.
                                 */
                                 .setMode(CUDNN_POINTWISE_BINARY_SELECT)
                                 .setComputeType(dpct::library_data_t::real_float)
                                 .build();
        std::cout << selectionDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1036: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1037: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(afterConvTensor)
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:1038: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(afterScaleTensor)
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:1039: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(afterBiasTensor)
                          .setyDesc(afterActivationTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create a Gen_Index Node.
        /*
        DPCT1007:1040: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto genIndex_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(afterActivationTensor)
                               .setyDesc(genIndexTensor)
                               .setpwDesc(genIndexDesc)
                               .build();
        std::cout << genIndex_op.describe() << std::endl;

        // Create a LessThan Node.
        /*
        DPCT1007:1041: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto lessThan_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(genIndexTensor)
                               .setbDesc(thresholdTopTensor)
                               .setyDesc(maskTopTensor)
                               .setpwDesc(lessThanDesc)
                               .build();
        std::cout << lessThan_op.describe() << std::endl;

        // Create a GreaterThan Node.
        /*
        DPCT1007:1042: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto greaterThan_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                  .setxDesc(genIndexTensor)
                                  .setbDesc(thresholdBottomTensor)
                                  .setyDesc(maskBottomTensor)
                                  .setpwDesc(greaterThanDesc)
                                  .build();
        std::cout << greaterThan_op.describe() << std::endl;

        // Create a LogicalOr Node.
        /*
        DPCT1007:1043: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto logicalOr_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                .setxDesc(maskTopTensor)
                                .setbDesc(maskBottomTensor)
                                .setyDesc(maskTensor)
                                .setpwDesc(logicalOrDesc)
                                .build();
        std::cout << logicalOr_op.describe() << std::endl;

        // Create a Binary_Selection Node.
        /*
        DPCT1007:1044: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto selection_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                .setxDesc(afterConvTensor)
                                .setbDesc(afterActivationTensor)
                                .settDesc(maskTensor)
                                .setyDesc(yTensor)
                                .setpwDesc(selectionDesc)
                                .build();
        std::cout << selection_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 9> ops = {&conv_op,
                                                               &scale_op,
                                                               &bias_op,
                                                               &act_op,
                                                               &genIndex_op,
                                                               &lessThan_op,
                                                               &greaterThan_op,
                                                               &logicalOr_op,
                                                               &selection_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // How many engines support this operation graph ?
        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrS, devPtrB, devPtrTopThreshold, devPtrBottomThreshold};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'b', 't', 'u'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1045: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1046: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
#endif

    } catch (cudnn_frontend::cudnnException& e) {
        if (e.getCudnnStatus() == 3007) {
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_conv_scale_bias_relu_int8(int64_t* x_dim,
                              int64_t* w_dim,
                              int64_t* y_dim,
                              int64_t* s_dim,
                              int64_t* b_dim,
                              int convDim,
                              int64_t* conv_padA,
                              int64_t* conv_dilationA,
                              int64_t* conv_strideA,
                              void* devPtrX,
                              void* devPtrW,
                              void* devPtrY,
                              void* devPtrS,
                              void* devPtrB) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        if (check_device_arch_newer_than("turing") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_conv_scale_bias_relu_int8: Sample requires Turing or above GPU");
        }

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dpct::library_data_t::real_int8)
                           .build();
        generateStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_int8)
                           .build();
        generateStrides(s_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, stride)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_float)
                           .build();

        generateStrides(b_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, stride)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_float)
                           .build();

        generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dpct::library_data_t::real_int32)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, stride)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(dpct::library_data_t::real_float)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dpct::library_data_t::real_float)
                                   .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, stride)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_int8)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_int32)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             /*
                             DPCT1007:1047: Migration of CUDNN_POINTWISE_MUL is not supported.
                             */
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:1048: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(dpct::library_data_t::real_float)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1049: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1050: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1051: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(conv_op.getOutputTensor())
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:1052: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(scale_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:1053: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &scale_op, &bias_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // How many engines support this operation graph ?
        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrS, devPtrB};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'b'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(5, data_ptrs)
                               .setUids(5, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1054: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1055: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        // this example is only for Turing and later cards
        /*
        DPCT1005:1056: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for Turing GPUs" << std::endl;
        } else {
#if (CUDNN_VERSION == 8600)
            if (prop.major == 9) {
                std::cout << "Hopper GPUs does not have int8 fused operations support yet\n";
                return;
            }
#endif
#if (CUDNN_VERSION >= 8300)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif
        }
    }
}

void
run_pool_scale_bias_relu_int8(int64_t* x_dim,
                              int64_t* y_dim,
                              int64_t* s_dim,
                              int64_t* b_dim,
                              void* devPtrX,
                              void* devPtrY,
                              void* devPtrS,
                              void* devPtrB,
                              dpct::library_data_t compType,
                              int const nanOpt,
                              cudnn_frontend::ResampleMode_t const mode,
                              cudnn_frontend::PaddingMode_t const padding_mode,
                              int64_t nbSpatialDims,
                              double alpha,
                              double beta,
                              int64_t* windowDimA,
                              int64_t* prePaddingA,
                              int64_t* postPaddingA,
                              int64_t* strideA) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    (void)nbSpatialDims;
    (void)alpha;
    (void)beta;
    (void)windowDimA;
    (void)prePaddingA;
    (void)postPaddingA;
    (void)strideA;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t strideTensor[4];
        generateStrides(x_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, strideTensor)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dpct::library_data_t::real_int8)
                           .build();
        generateStrides(s_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, s_dim)
                           .setStride(4, strideTensor)
                           .setId('s')
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_float)
                           .build();

        generateStrides(b_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto bTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, b_dim)
                           .setStride(4, strideTensor)
                           .setId('b')
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_float)
                           .build();

        generateStrides(y_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterPoolTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, strideTensor)
                                   .setId('A')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(compType)
                                   .build();
        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, strideTensor)
                                    .setId('B')  // after scale
                                    .setAlignment(16)
                                    .setVirtual()
                                    .setDataType(compType)
                                    .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, strideTensor)
                                   .setId('C')  // after bias
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(compType)
                                   .build();
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStride(4, strideTensor)
                           .setId('y')  // output
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_int8)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << bTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << afterPoolTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << afterScaleTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setNanPropagation(nanOpt)
                            .setResampleMode(mode)
                            .setPaddingMode(padding_mode)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setSpatialStride(nbSpatialDims, strideA)
                            .setPrePadding(nbSpatialDims, prePaddingA)
                            .setPostPadding(nbSpatialDims, postPaddingA)
                            .build();
        std::cout << "Initialized Pool Desc" << std::endl;
        std::cout << poolDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc =
            /*
            DPCT1007:1057: Migration of CUDNN_POINTWISE_MUL is not supported.
            */
            cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_MUL).setComputeType(compType).build();
        std::cout << "Initialized Scale Desc" << std::endl;
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc =
            /*
            DPCT1007:1058: Migration of CUDNN_POINTWISE_ADD is not supported.
            */
            cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_ADD).setComputeType(compType).build();
        std::cout << "Initialized Bias Desc" << std::endl;
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc =
            /*
            DPCT1007:1059: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
            */
            cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_RELU_FWD).setComputeType(compType).build();
        std::cout << "Initialized Activation Desc" << std::endl;
        std::cout << actDesc.describe() << std::endl;

#if (CUDNN_VERSION >= 8500)
        // Create a Resample Node
        /*
        DPCT1007:1060: Migration of CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR is not supported.
        */
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setyDesc(afterPoolTensor)
                           .setResampleDesc(poolDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << pool_op.describe() << std::endl;
#endif
        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1061: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
#if (CUDNN_VERSION >= 8500)
                            .setxDesc(pool_op.getOutputTensor())
#endif
                            .setbDesc(sTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:1062: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(scale_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:1063: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setyDesc(yTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

#if (CUDNN_VERSION >= 8500)
        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 4> ops = {&pool_op, &scale_op, &bias_op, &act_op};
#else
        std::array<cudnn_frontend::Operation const*, 3> ops = {&scale_op, &bias_op, &act_op};
#endif
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // How many engines support this operation graph ?
        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        // Create the variant pack and associate with the data pointers
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrS, devPtrB};
        int64_t uids[]    = {'x', 'y', 's', 'b'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // Trigger the execute operation
        /*
        DPCT1007:1064: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }
        /*
        DPCT1027:1065: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
        std::cout << "EXECUTE SUCCESS" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        std::cout << "Sample not executed for cuDNN version " << CUDNN_VERSION << std::endl;
        // this example is only for Ampere cards
        /*
        DPCT1005:1066: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
#if (CUDNN_VERSION >= 8500)
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
#endif
        }
    }
}

void
run_matmul_bias_gelu(int64_t* a_dim,
                     int64_t* b_dim,
                     int64_t* c_dim,
                     int64_t* z_dim,
                     dpct::library_data_t dataType,
                     void* devPtrA,
                     void* devPtrB,
                     void* devPtrC,
                     void* devPtrZ,
                     void* devPtrAfterZ) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("ampere") == false && dataType == dpct::library_data_t::real_float) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_matmul_bias_gelu: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[3];
        // the intension is to compute stride for a [1, M, K] matrix with K in the inner most dimension, and
        // CUDNN_TENSOR_NCHW is a borrowed notation
        generateStrides(a_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto aMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, a_dim)
                                 .setStride(3, stride)
                                 .setId('a')
                                 .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                 .setDataType(dataType)
                                 .build();
        generateStrides(b_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto bMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, b_dim)
                                 .setStride(3, stride)
                                 .setId('b')
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();

        generateStrides(z_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto biasTensor = cudnn_frontend::TensorBuilder()
                              .setDim(3, z_dim)
                              .setStride(3, stride)
                              .setId('z')
                              .setAlignment(16)
                              .setDataType(dataType)
                              .build();

        generateStrides(c_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto afterMatMulTensor = cudnn_frontend::TensorBuilder()
                                     .setDim(3, c_dim)
                                     .setStride(3, stride)
                                     .setId('A')  // after matmul
                                     .setAlignment(16)
                                     .setVirtual()
                                     .setDataType(dataType)
                                     .build();
        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(3, c_dim)
                                   .setStride(3, stride)
                                   .setId('B')  // after bias
                                   .setAlignment(16)
                                   .setDataType(dataType)
                                   .build();
        auto outputTensor = cudnn_frontend::TensorBuilder()
                                .setDim(3, c_dim)
                                .setStride(3, stride)
                                .setId('c')  // output after gelu
                                .setAlignment(16)
                                .setDataType(dataType)
                                .build();

        std::cout << aMatrixTensor.describe() << std::endl;
        std::cout << bMatrixTensor.describe() << std::endl;
        std::cout << biasTensor.describe() << std::endl;
        std::cout << afterMatMulTensor.describe() << std::endl;
        std::cout << afterBiasTensor.describe() << std::endl;
        std::cout << outputTensor.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:1067: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(dpct::library_data_t::real_float)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
#if (CUDNN_VERSION >= 8500)
                           /*
                           DPCT1007:1068: Migration of CUDNN_POINTWISE_GELU_APPROX_TANH_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_GELU_APPROX_TANH_FWD)
#else
                           .setMode(CUDNN_POINTWISE_GELU_FWD)
#endif
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the matmul desc
        auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
        std::cout << matmulDesc.describe() << std::endl;

        // Create a matmul Node
        /*
        DPCT1007:1069: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
        */
        auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                             .setaMatDesc(aMatrixTensor)
                             .setbMatDesc(bMatrixTensor)
                             .setcMatDesc(afterMatMulTensor)
                             .setmatmulDesc(matmulDesc)
                             .build();
        std::cout << matmul_op.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:1070: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(matmul_op.getOutputTensor())
                           .setbDesc(biasTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:1071: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setyDesc(outputTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is matmul bias activation
        std::array<cudnn_frontend::Operation const*, 3> ops = {&matmul_op, &bias_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrA, devPtrB, devPtrC, devPtrZ, devPtrAfterZ};
        int64_t uids[]    = {'a', 'b', 'c', 'z', 'B'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(5, data_ptrs)
                               .setUids(5, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1072: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1073: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere cards
        /*
        DPCT1005:1074: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3000 || e.getCudnnStatus() == 3007)) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
            CHECK(false);
#endif
        }
    }
}

void
run_conv_drelu(int64_t* x_dim,
               int64_t* pad,
               int64_t* convstride,
               int64_t* dilation,
               int64_t* w_dim,
               int64_t* y_dim,
               dpct::library_data_t dataType,
               void* dev_ptr_x,
               void* dev_ptr_w,
               void* dev_ptr_y,
               void* dev_ptr_bwd_act_x) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        int convDim = 2;

        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_conv_drelu: Sample requires Ampere or above GPU");
        }
        int64_t x_id         = 101;
        int64_t w_id         = 102;
        int64_t bwd_act_x_id = 201;
        int64_t y_id         = 301;

        int64_t after_conv_id = 1001;

        int64_t x_stride_padded[4];
        int64_t y_stride_padded[4];
        int64_t w_stride_padded[4];

        generateStrides(w_dim, w_stride_padded, 4, dpct::dnnl::memory_format_tag::nhwc);
        generateStrides(x_dim, x_stride_padded, 4, dpct::dnnl::memory_format_tag::nhwc);
        generateStrides(y_dim, y_stride_padded, 4, dpct::dnnl::memory_format_tag::nhwc);

        auto x_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, x_dim)
                            .setStride(4, x_stride_padded)
                            .setId(x_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto w_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, w_dim)
                            .setStride(4, w_stride_padded)
                            .setId(w_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto after_conv_tensor = cudnn_frontend::TensorBuilder()
                                     .setDim(4, y_dim)
                                     .setStride(4, y_stride_padded)
                                     .setId(after_conv_id)
                                     .setAlignment(4)
                                     .setVirtual()
                                     .setDataType(dpct::library_data_t::real_float)
                                     .build();

        auto bwd_act_x_tensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStride(4, y_stride_padded)
                                    .setId(bwd_act_x_id)
                                    .setAlignment(4)
                                    .setDataType(dataType)
                                    .build();

        auto after_activation_tensor = cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim)
                                           .setStride(4, y_stride_padded)
                                           .setId(y_id)
                                           .setAlignment(4)
                                           .setDataType(dataType)
                                           .build();

        std::cout << x_tensor.describe() << std::endl;
        std::cout << w_tensor.describe() << std::endl;
        std::cout << after_conv_tensor.describe() << std::endl;
        std::cout << bwd_act_x_tensor.describe() << std::endl;
        std::cout << after_activation_tensor.describe() << std::endl;

        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        /*
        DPCT1007:1075: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(x_tensor)
                           .setwDesc(w_tensor)
                           .setyDesc(after_conv_tensor)
                           .setcDesc(convDesc)
                           .setAlpha(1.0f)
                           .setBeta(0.0f)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1076: Migration of CUDNN_POINTWISE_RELU_BWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_BWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        /*
        DPCT1007:1077: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setdyDesc(after_conv_tensor)
                          .setxDesc(bwd_act_x_tensor)
                          .setdxDesc(after_activation_tensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {dev_ptr_x, dev_ptr_w, dev_ptr_bwd_act_x, dev_ptr_y};
        int64_t uids[]    = {x_id, w_id, bwd_act_x_id, y_id};

        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        /*
        DPCT1007:1078: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1079: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        if (e.getCudnnStatus() == 3007) {
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;

        CHECK(false);
    }
}

void
run_dgrad_drelu(int64_t* dx_dim,
                int64_t* pad,
                int64_t* convstride,
                int64_t* dilation,
                int64_t* w_dim,
                int64_t* dy_dim,
                dpct::library_data_t dataType,
                void* dev_ptr_dx,
                void* dev_ptr_w,
                void* dev_ptr_dy,
                void* dev_ptr_bwd_act_x) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        int convDim = 2;
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_dgrad_drelu: Sample requires Ampere or above GPU");
        }
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        int64_t dx_id        = 101;
        int64_t w_id         = 102;
        int64_t bwd_act_x_id = 201;
        int64_t dy_id        = 301;

        int64_t after_dgrad_id = 1001;

        int64_t dx_stride[4];
        int64_t dy_stride[4];
        int64_t w_stride[4];

        generateStrides(w_dim, w_stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        generateStrides(dx_dim, dx_stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        generateStrides(dy_dim, dy_stride, 4, dpct::dnnl::memory_format_tag::nhwc);

        auto after_dgrad_dx_tensor = cudnn_frontend::TensorBuilder()
                                         .setDim(4, dx_dim)
                                         .setStride(4, dx_stride)
                                         .setId(after_dgrad_id)
                                         .setAlignment(4)
                                         .setVirtual()
                                         .setDataType(dataType)
                                         .build();

        auto w_tensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, w_dim)
                            .setStride(4, w_stride)
                            .setId(w_id)
                            .setAlignment(4)
                            .setDataType(dataType)
                            .build();

        auto dy_tensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, dy_dim)
                             .setStride(4, dy_stride)
                             .setId(dy_id)
                             .setAlignment(4)
                             .setDataType(dataType)
                             .build();

        auto bwd_act_x_tensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, dx_dim)
                                    .setStride(4, dx_stride)
                                    .setId(bwd_act_x_id)
                                    .setAlignment(4)
                                    .setDataType(dataType)
                                    .build();

        auto after_bwd_activation_dx_tensor = cudnn_frontend::TensorBuilder()
                                                  .setDim(4, dx_dim)
                                                  .setStride(4, dx_stride)
                                                  .setId(dx_id)
                                                  .setAlignment(4)
                                                  .setDataType(dataType)
                                                  .build();

        std::cout << after_dgrad_dx_tensor.describe() << std::endl;
        std::cout << w_tensor.describe() << std::endl;
        std::cout << dy_tensor.describe() << std::endl;
        std::cout << bwd_act_x_tensor.describe() << std::endl;
        std::cout << after_bwd_activation_dx_tensor.describe() << std::endl;

        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        /*
        DPCT1007:1080: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                           .setdyDesc(dy_tensor)
                           //    .setyDesc(dy_tensor)
                           .setwDesc(w_tensor)
                           .setdxDesc(after_dgrad_dx_tensor)
                           //    .setxDesc(after_dgrad_dx_tensor)
                           .setcDesc(convDesc)
                           .setAlpha(1.0f)
                           .setBeta(0.0f)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1081: Migration of CUDNN_POINTWISE_RELU_BWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_BWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;

        /*
        DPCT1007:1082: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setdyDesc(after_dgrad_dx_tensor)
                          .setxDesc(bwd_act_x_tensor)
                          .setdxDesc(after_bwd_activation_dx_tensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;

        std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {dev_ptr_dx, dev_ptr_w, dev_ptr_bwd_act_x, dev_ptr_dy};
        int64_t uids[]    = {dx_id, w_id, bwd_act_x_id, dy_id};

        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        /*
        DPCT1007:1083: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1084: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        if (e.getCudnnStatus() == 3007) {
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_matmul_dgelu_dbias(const int64_t* dy_dim,
                       const int64_t* w_dim,
                       const int64_t* dx_dim,
                       const int64_t* dbias_dim,
                       dpct::library_data_t dataType,
                       void* dev_ptr_dy,
                       void* dev_ptr_w,
                       void* dev_ptr_bwd_act_x,
                       void* dev_ptr_dx,
                       void* dev_ptr_dbias) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_matmul_dgelu_dbias: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[3];

        // Use the following UIDs for tensors
        int64_t dy_uid        = 101;
        int64_t w_uid         = 102;
        int64_t bwd_act_x_uid = 103;
        int64_t dx_uid        = 104;
        int64_t dbias_uid     = 105;

        // Create tensor descriptor for DY matrix
        generateStrides(dy_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto dyMatrixTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(3, dy_dim)
                                  .setStride(3, stride)
                                  .setId(dy_uid)
                                  .setAlignment(16)
                                  .setDataType(dataType)
                                  .build();
        std::cout << dyMatrixTensor.describe() << std::endl;

        // Create tensor descriptor for weight matrix
        generateStrides(w_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto wMatrixTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(3, w_dim)
                                 .setStride(3, stride)
                                 .setId(w_uid)
                                 .setAlignment(16)
                                 .setDataType(dataType)
                                 .build();
        std::cout << wMatrixTensor.describe() << std::endl;

        // Create tensor descriptor for dx matrix
        generateStrides(dx_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto dataGrad1MatrixTensor = cudnn_frontend::TensorBuilder()
                                         .setDim(3, dx_dim)
                                         .setStride(3, stride)
                                         .setId('X')
                                         .setAlignment(16)
                                         .setDataType(dataType)
                                         .setVirtual(true)
                                         .build();
        std::cout << dataGrad1MatrixTensor.describe() << std::endl;

        // Create tensor descriptor for geluInput matrix
        generateStrides(dx_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto geluInputMatrixTensor = cudnn_frontend::TensorBuilder()
                                         .setDim(3, dx_dim)
                                         .setStride(3, stride)
                                         .setId(bwd_act_x_uid)
                                         .setAlignment(16)
                                         .setDataType(dataType)
                                         .build();
        std::cout << geluInputMatrixTensor.describe() << std::endl;

        // Create tensor descriptor for output of backwardGelu matrix
        generateStrides(dx_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto backwardGeluMatrixTensor = cudnn_frontend::TensorBuilder()
                                            .setDim(3, dx_dim)
                                            .setStride(3, stride)
                                            .setId(dx_uid)
                                            .setAlignment(16)
                                            .setDataType(dataType)
                                            .build();
        std::cout << backwardGeluMatrixTensor.describe() << std::endl;

        // Create tensor descriptor for output of biasGrad(reduction) matrix
        generateStrides(dbias_dim, stride, 3, dpct::dnnl::memory_format_tag::nchw);
        auto backwardBiasMatrixTensor = cudnn_frontend::TensorBuilder()
                                            .setDim(3, dbias_dim)
                                            .setStride(3, stride)
                                            .setId(dbias_uid)
                                            .setAlignment(16)
                                            .setDataType(dpct::library_data_t::real_float)
                                            .build();
        std::cout << backwardBiasMatrixTensor.describe() << std::endl;

        auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
        std::cout << matmulDesc.describe() << std::endl;

        auto geluDesc = cudnn_frontend::PointWiseDescBuilder()
#if (CUDNN_VERSION >= 8500)
                            /*
                            DPCT1007:1085: Migration of CUDNN_POINTWISE_GELU_APPROX_TANH_BWD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_GELU_APPROX_TANH_BWD)
#else
                            .setMode(CUDNN_POINTWISE_GELU_BWD)
#endif
                            .setComputeType(dpct::library_data_t::real_float)
                            .build();
        std::cout << geluDesc.describe() << std::endl;

        // Define the reduction descriptor
        auto reductionDesc = cudnn_frontend::ReductionDescBuilder()
                                 .setComputeType(dpct::library_data_t::real_float)
                                 .setReductionOp(dpct::dnnl::reduction_op::sum)
                                 .build();
        std::cout << reductionDesc.describe() << std::endl;

        // Create a matmul Node for Dgrad
        /*
        DPCT1007:1086: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
        */
        auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(dyMatrixTensor)
                            .setbMatDesc(wMatrixTensor)
                            .setcMatDesc(dataGrad1MatrixTensor)
                            .setmatmulDesc(matmulDesc)
                            .build();
        std::cout << matmulOp.describe() << std::endl;

        // Create a matmul Node for dGeLU
        /*
        DPCT1007:1087: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto geluOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setdyDesc(matmulOp.getOutputTensor())
                          .setxDesc(geluInputMatrixTensor)
                          .setdxDesc(backwardGeluMatrixTensor)
                          .setpwDesc(geluDesc)
                          .build();
        std::cout << geluOp.describe() << std::endl;

        // Create a reduction add Node.
        /*
        DPCT1007:1088: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
        */
        auto reduction_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(backwardGeluMatrixTensor)
                                .setyDesc(backwardBiasMatrixTensor)
                                .setreductionDesc(reductionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph.
        std::array<cudnn_frontend::Operation const*, 3> ops = {&matmulOp, &geluOp, &reduction_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {dev_ptr_dy, dev_ptr_w, dev_ptr_dx, dev_ptr_bwd_act_x, dev_ptr_dbias};
        int64_t uids[]    = {dy_uid, w_uid, dx_uid, bwd_act_x_uid, dbias_uid};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(5, data_ptrs)
                               .setUids(5, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1089: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1090: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere cards
        /*
        DPCT1005:1091: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3000 || e.getCudnnStatus() == 3007)) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }
}

void
run_conv_reduction(int64_t* x_dim,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   int64_t* r_dim,
                   dpct::library_data_t dataType,
                   int convDim,
                   int64_t* conv_padA,
                   int64_t* conv_dilationA,
                   int64_t* conv_strideA,
                   void* devPtrX,
                   void* devPtrW,
                   void* devPtrR) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_conv_reduction: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        generateStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, w_dim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        generateStrides(r_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto rTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, r_dim)
                           .setStride(4, stride)
                           .setId('r')  // output
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_float)
                           .build();

        generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStride(4, stride)
                                   .setId('y')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dataType)
                                   .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << rTensor.describe() << std::endl;

        std::cout << afterConvTensor.describe() << std::endl;

        // Define the reduction descriptor
        auto redunctionDesc = cudnn_frontend::ReductionDescBuilder()
                                  .setComputeType(dpct::library_data_t::real_float)
                                  .setReductionOp(dpct::dnnl::reduction_op::sum)
                                  .build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1092: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a reduction add Node.
        /*
        DPCT1007:1093: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
        */
        auto reduction_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(conv_op.getOutputTensor())
                                .setyDesc(rTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution reduction add
        std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &reduction_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrR};
        int64_t uids[]    = {'x', 'w', 'r'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1094: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1095: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnn_frontend::cudnnException& e) {
        if (e.getCudnnStatus() == 3007) {
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

dpct::err1
run_bn_conv_gen_stat(int64_t* xTensorDim,
                     int64_t* wTensorDim,
                     int64_t* yTensorDim,
                     int64_t* scaleTensorDim,
                     int convDim,
                     int64_t* conv_padA,
                     int64_t* conv_dilationA,
                     int64_t* conv_strideA,
                     void* XdevPtr,
                     void* WdevPtr,
                     void* YdevPtr,
                     void* scaledevPtr,
                     void* biasdevPtr,
                     void* sumdevPtr,
                     void* sqSumdevPtr) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(xTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, xTensorDim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dpct::library_data_t::real_half)
                           .build();

        auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                                    .setDim(4, xTensorDim)
                                    .setStride(4, stride)
                                    .setId('d')
                                    .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                    .setDataType(dpct::library_data_t::real_float)
                                    .setVirtual()
                                    .build();

        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, xTensorDim)
                                   .setStride(4, stride)
                                   .setId('e')
                                   .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                   .setDataType(dpct::library_data_t::real_float)
                                   .setVirtual()
                                   .build();

        auto afterReluTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, xTensorDim)
                                   .setStride(4, stride)
                                   .setId('f')
                                   .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                   .setDataType(dpct::library_data_t::real_float)
                                   .setVirtual()
                                   .build();

        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto scaleTensor = cudnn_frontend::TensorBuilder()
                               .setDim(4, scaleTensorDim)
                               .setStride(4, stride)
                               .setId('s')
                               .setAlignment(16)
                               .setDataType(dpct::library_data_t::real_half)
                               .build();

        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto biasTensor = cudnn_frontend::TensorBuilder()
                              .setDim(4, scaleTensorDim)
                              .setStride(4, stride)
                              .setId('b')
                              .setAlignment(16)
                              .setDataType(dpct::library_data_t::real_half)
                              .build();
        generateStrides(wTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, wTensorDim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_half)
                           .build();

        generateStrides(yTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, yTensorDim)
                           .setStride(4, stride)
                           .setId('y')  // after conv
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_half)
                           .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             /*
                             DPCT1007:1096: Migration of CUDNN_POINTWISE_MUL is not supported.
                             */
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            /*
                            DPCT1007:1097: Migration of CUDNN_POINTWISE_ADD is not supported.
                            */
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setComputeType(dpct::library_data_t::real_float)
                            .build();
        std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1098: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .setReluLowerClipSlope(0.01)  // leaky relu
                           .build();
        std::cout << actDesc.describe() << std::endl;
        std::cout << "Creating OPs " << std::endl;
        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1099: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(xTensor)
                            .setbDesc(scaleTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a Bias Node.
        /*
        DPCT1007:1100: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(afterScaleTensor)
                           .setbDesc(biasTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        /*
        DPCT1007:1101: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(afterBiasTensor)
                          .setyDesc(afterReluTensor)
                          .setpwDesc(actDesc)
                          .build();
        std::cout << act_op.describe() << std::endl;
        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1102: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(afterReluTensor)
                           .setwDesc(wTensor)
                           .setyDesc(yTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto sumTensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, scaleTensorDim)
                             .setStride(4, stride)
                             .setId('u')
                             .setAlignment(16)
                             .setDataType(dpct::library_data_t::real_float)
                             .build();

        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto sqsumTensor = cudnn_frontend::TensorBuilder()
                               .setDim(4, scaleTensorDim)
                               .setStride(4, stride)
                               .setId('v')
                               .setAlignment(16)
                               .setDataType(dpct::library_data_t::real_float)
                               .build();

        // Create a genstats node
        /*
        DPCT1007:1104: Migration of CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR is not supported.
        */
        auto genstat_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR)
                              .setxDesc(yTensor)
                              .setComputeType(dpct::library_data_t::real_float)
                              /*
                              DPCT1007:1103: Migration of CUDNN_GENSTATS_SUM_SQSUM is not supported.
                              */
                              .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
                              .setSumDesc(sumTensor)
                              .setSqSumDesc(sqsumTensor)
                              .build();
        std::cout << genstat_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is scale bias Relu conv gen_stats
        std::array<cudnn_frontend::Operation const*, 5> ops = {&scale_op, &bias_op, &conv_op, &act_op, &genstat_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        cudnn_frontend::ManagedOpaqueDescriptor plan_desc = nullptr;
        int64_t workspace_size                            = 0;
        dpct::err1 st                                     = 0;
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
                st = e.getCudnnStatus();
                continue;
            }
        }
        if (plan_desc == nullptr) {
            std::cout << "No plan found implementing the operation graph" << std::endl;
            return st;
        }

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        void* data_ptrs[] = {XdevPtr, WdevPtr, YdevPtr, scaledevPtr, biasdevPtr, sumdevPtr, sqSumdevPtr};
        int64_t uids[]    = {'x', 'w', 'y', 's', 'b', 'u', 'v'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        dpct::err1 status =
            /*
            DPCT1007:1105: Migration of cudnnBackendExecute is not supported.
            */
            cudnnBackendExecute(handle_, plan_desc->get_backend_descriptor(), variantPack.get_raw_desc());

        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);

        return status;

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere cards
        /*
        DPCT1005:1106: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && e.getCudnnStatus() == 3000) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
            CHECK(false);
#endif
        }
        return 0;
    }
}

void
run_bn_finalize(int64_t* perChannelSum,
                int64_t* epsilon,

                void* YSumdevPtr,
                void* YSqSumdevPtr,
                void* scaledevPtr,
                void* biasdevPtr,
                void* in_meandevPtr,
                void* in_vardevPtr,
                void* out_meandevPtr,
                void* out_vardevPtr,
                void* saved_meandevPtr,
                void* saved_inv_vardevPtr,
                void* eq_scaledevPtr,
                void* eq_biasdevPtr,

                double epsilon_val,
                double exponential_decay_factor,
                int64_t accumCnt_val) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(perChannelSum, stride, 4, dpct::dnnl::memory_format_tag::nhwc);

        auto tensor_create = [&stride, &perChannelSum](dpct::library_data_t type, int64_t id) {
            return cudnn_frontend::TensorBuilder()
                .setDim(4, perChannelSum)
                .setStride(4, stride)
                .setId(id)
                .setAlignment(16)
                .setDataType(type)
                .build();
        };

        auto sumTensor         = tensor_create(dpct::library_data_t::real_float, 100);
        auto sqSumTensor       = tensor_create(dpct::library_data_t::real_float, 101);
        auto scaleTensor       = tensor_create(dpct::library_data_t::real_float, 102);
        auto biasTensor        = tensor_create(dpct::library_data_t::real_float, 103);
        auto inMeanTensor      = tensor_create(dpct::library_data_t::real_float, 104);
        auto inVarTensor       = tensor_create(dpct::library_data_t::real_float, 105);
        auto outMeanTensor     = tensor_create(dpct::library_data_t::real_float, 106);
        auto outVarTensor      = tensor_create(dpct::library_data_t::real_float, 107);
        auto savedMeanTensor   = tensor_create(dpct::library_data_t::real_float, 108);
        auto savedInvVarTensor = tensor_create(dpct::library_data_t::real_float, 109);
        auto outEqScaleTensor  = tensor_create(dpct::library_data_t::real_float, 200);
        auto outEqBiasTensor   = tensor_create(dpct::library_data_t::real_float, 201);

        int64_t epsilon_stride[4];
        generateStrides(epsilon, epsilon_stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto scalar_tensor_create = [&epsilon_stride, &epsilon](dpct::library_data_t type, int64_t id) {
            return cudnn_frontend::TensorBuilder()
                .setDim(4, epsilon)
                .setStride(4, epsilon_stride)
                .setId(id)
                .setAlignment(16)
                .setDataType(type)
                .setByValue(true)
                .build();
        };

        auto epsilonTensor  = scalar_tensor_create(dpct::library_data_t::real_double, 300);
        auto expDecayTensor = scalar_tensor_create(dpct::library_data_t::real_double, 301);
        /*
        DPCT1007:1107: Migration of CUDNN_DATA_INT64 is not supported.
        */
        auto accumCountTensor = scalar_tensor_create(CUDNN_DATA_INT64, 302);

        // Create a Finalize node
        auto finalize_stat_op =
            /*
            DPCT1007:1109: Migration of CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR is not supported.
            */
            cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
                .setComputeType(dpct::library_data_t::real_float)
                /*
                DPCT1007:1108: Migration of CUDNN_BN_FINALIZE_STATISTICS_TRAINING is not supported.
                */
                .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
                .setSumDesc(sumTensor)
                .setSqSumDesc(sqSumTensor)
                .setScaleAndBias(scaleTensor, biasTensor)
                .setEqScaleAndBias(outEqScaleTensor, outEqBiasTensor)
                .setPrevRunningMeanAndVar(inMeanTensor, inVarTensor)
                .setNextRunningMeanAndVar(outMeanTensor, outVarTensor)
                .setSavedMeanAndInvVar(savedMeanTensor, savedInvVarTensor)
                .setEpsilonTensor(epsilonTensor)
                .setAccumCountTensor(accumCountTensor)
                .setExpDecayFactorTensor(expDecayTensor)
                .build();

        std::array<cudnn_frontend::Operation const*, 1> ops = {&finalize_stat_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan_builder = [&filtered_configs, &opGraph, &dpct::dnnl::engine_ext]() {
            for (size_t i = 0; i < filtered_configs.size(); i++) {
                try {
                    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                    .setHandle(handle_)
                                    .setEngineConfig(filtered_configs[i], opGraph.getTag())
                                    .build();
                    return plan;
                } catch (cudnn_frontend::cudnnException&) {
                    continue;
                }
            }
            return cudnn_frontend::ExecutionPlanBuilder()
                .setHandle(handle_)
                .setEngineConfig(filtered_configs[0], opGraph.getTag())
                .build();
        };

        REQUIRE(filtered_configs.size() > 0);
        auto plan = plan_builder();
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        void* data_ptrs[15] = {YSumdevPtr,
                               YSqSumdevPtr,
                               scaledevPtr,
                               biasdevPtr,
                               in_meandevPtr,
                               in_vardevPtr,
                               out_meandevPtr,
                               out_vardevPtr,
                               saved_meandevPtr,
                               saved_inv_vardevPtr,
                               eq_scaledevPtr,
                               eq_biasdevPtr,
                               &epsilon_val,
                               &exponential_decay_factor,
                               &accumCnt_val};
        int64_t uids[15]    = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 200, 201, 300, 301, 302};
        auto variantPack    = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(15, data_ptrs)
                               .setUids(15, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1110: Migration of cudnnBackendExecute is not supported.
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

        std::cout << "BN Finalize run completed successfully" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
#if (CUDNN_VERSION >= 8400)
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
#endif
    }
}

dpct::err1
run_dsbar(int64_t* Y_dim,
          int64_t* scaleTensorDim,
          void* RP_YdevPtr,
          void* RP_scaleDevPtr,
          void* RP_biasDevPtr,
          void* DP_YdevPtr,
          void* DP_scaleDevPtr,
          void* DP_biasDevPtr,
          void* YdevPtr,
          dpct::library_data_t op_data_type) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;

    try {
        // Create a handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Create tensor descriptors
        int64_t stride[4];

        // RP_Y tensor
        generateStrides(Y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto RP_yTensor = cudnn_frontend::TensorBuilder()
                              .setDim(4, Y_dim)
                              .setStride(4, stride)
                              .setId('y')
                              .setAlignment(16)  // 16 byte alignment
                              .setDataType(dpct::library_data_t::real_half)
                              .build();

        // RP_scale tensor
        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto RP_scaleTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, scaleTensorDim)
                                  .setStride(4, stride)
                                  .setId('s')
                                  .setAlignment(16)  // 16 byte alignment
                                  .setDataType(dpct::library_data_t::real_float)
                                  .build();

        // After RP scale tensor (RP_yTensor * RP_scaleTensor)
        generateStrides(Y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto RP_afterScaleTensor = cudnn_frontend::TensorBuilder()
                                       .setDim(4, Y_dim)
                                       .setStride(4, stride)
                                       .setId('d')
                                       .setVirtual()
                                       .setAlignment(16)  // 16 byte alignment
                                       .setDataType(dpct::library_data_t::real_float)
                                       .build();

        // RP_bias tensor
        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto RP_biasTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(4, scaleTensorDim)
                                 .setStride(4, stride)
                                 .setId('b')
                                 .setAlignment(16)  // 16 byte alignment
                                 .setDataType(dpct::library_data_t::real_float)
                                 .build();

        // After RP bias tensor (RP_afterScaleTensor + RP_biasTensor)
        generateStrides(Y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto RP_afterBiasTensor = cudnn_frontend::TensorBuilder()
                                      .setDim(4, Y_dim)
                                      .setStride(4, stride)
                                      .setId('e')
                                      .setVirtual()
                                      .setAlignment(16)  // 16 byte alignment
                                      .setDataType(dpct::library_data_t::real_float)
                                      .build();

        // DP_Y tensor
        generateStrides(Y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto DP_yTensor = cudnn_frontend::TensorBuilder()
                              .setDim(4, Y_dim)
                              .setStride(4, stride)
                              .setId('a')
                              .setAlignment(16)  // 16 byte alignment
                              .setDataType(dpct::library_data_t::real_half)
                              .build();

        // DP_scale tensor
        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto DP_scaleTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, scaleTensorDim)
                                  .setStride(4, stride)
                                  .setId('h')
                                  .setAlignment(16)  // 16 byte alignment
                                  .setDataType(dpct::library_data_t::real_float)
                                  .build();

        // After DP scale tensor (DP_yTensor * DP_scaleTensor)
        generateStrides(Y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto DP_afterScaleTensor = cudnn_frontend::TensorBuilder()
                                       .setDim(4, Y_dim)
                                       .setStride(4, stride)
                                       .setId('p')
                                       .setVirtual()
                                       .setAlignment(16)  // 16 byte alignment
                                       .setDataType(dpct::library_data_t::real_float)
                                       .build();

        // DP_bias tensor
        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto DP_biasTensor = cudnn_frontend::TensorBuilder()
                                 .setDim(4, scaleTensorDim)
                                 .setStride(4, stride)
                                 .setId('t')
                                 .setAlignment(16)  // 16 byte alignment
                                 .setDataType(dpct::library_data_t::real_float)
                                 .build();

        // After DP bias tensor (DP_afterScaleTensor + DP_biasTensor)
        generateStrides(Y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto DP_afterBiasTensor = cudnn_frontend::TensorBuilder()
                                      .setDim(4, Y_dim)
                                      .setStride(4, stride)
                                      .setId('n')
                                      .setVirtual()
                                      .setAlignment(16)  // 16 byte alignment
                                      .setDataType(dpct::library_data_t::real_float)
                                      .build();

        // After add RP_bias and DP_bias tensor (RP_afterBiasTensor + DP_afterBiasTensor)
        generateStrides(Y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterAddTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, Y_dim)
                                  .setStride(4, stride)
                                  .setId('m')
                                  .setVirtual()
                                  .setAlignment(16)  // 16 byte alignment
                                  .setDataType(dpct::library_data_t::real_float)
                                  .build();

        // Final output tensor after ReLU
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, Y_dim)
                           .setStride(4, stride)
                           .setId('f')
                           .setAlignment(16)  // 16 byte alignment
                           .setDataType(op_data_type)
                           .build();

        std::cout << RP_yTensor.describe() << std::endl;
        std::cout << DP_yTensor.describe() << std::endl;

        // Create the scale, add, and relu problems
        // Scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             /*
                             DPCT1007:1111: Migration of CUDNN_POINTWISE_MUL is not supported.
                             */
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Bias (add) descriptor
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1112: Migration of CUDNN_POINTWISE_ADD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << addDesc.describe() << std::endl;

        // ReLU descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           /*
                           DPCT1007:1113: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
                           */
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setComputeType(dpct::library_data_t::real_float)
                           .build();
        std::cout << actDesc.describe() << std::endl;
        std::cout << "Creating Operations now!" << std::endl;

        // Create RP scaling operation
        /*
        DPCT1007:1114: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto RP_scaleOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                              .setxDesc(RP_yTensor)
                              .setbDesc(RP_scaleTensor)
                              .setyDesc(RP_afterScaleTensor)
                              .setpwDesc(scaleDesc)
                              .build();
        std::cout << RP_scaleOp.describe() << std::endl;

        // Create RP bias operation
        /*
        DPCT1007:1115: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto RP_biasOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(RP_afterScaleTensor)
                             .setbDesc(RP_biasTensor)
                             .setyDesc(RP_afterBiasTensor)
                             .setpwDesc(addDesc)
                             .build();
        std::cout << RP_biasOp.describe() << std::endl;

        // Create DP scaling operation
        /*
        DPCT1007:1116: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto DP_scaleOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                              .setxDesc(DP_yTensor)
                              .setbDesc(DP_scaleTensor)
                              .setyDesc(DP_afterScaleTensor)
                              .setpwDesc(scaleDesc)
                              .build();
        std::cout << DP_scaleOp.describe() << std::endl;

        // Create DP bias operation
        /*
        DPCT1007:1117: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto DP_biasOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(DP_afterScaleTensor)
                             .setbDesc(DP_biasTensor)
                             .setyDesc(DP_afterBiasTensor)
                             .setpwDesc(addDesc)
                             .build();
        std::cout << DP_biasOp.describe() << std::endl;

        // Create add operation
        /*
        DPCT1007:1118: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto addOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(RP_afterBiasTensor)
                         .setbDesc(DP_afterBiasTensor)
                         .setyDesc(afterAddTensor)
                         .setpwDesc(addDesc)
                         .build();
        std::cout << addOp.describe() << std::endl;

        // Create ReLU operation
        /*
        DPCT1007:1119: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto actOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(afterAddTensor)
                         .setyDesc(yTensor)
                         .setpwDesc(actDesc)
                         .build();
        std::cout << actOp.describe() << std::endl;
        std::cout << "Creating operation graph now!" << std::endl;

        // Create an Operation Graph. In this case it is:
        // RP_scaleOp -> RP_biasOp -> DP_scaleOp -> DP_biasOp -> addOp -> reluOp
        std::array<cudnn_frontend::Operation const*, 6> ops = {
            &RP_scaleOp, &RP_biasOp, &DP_scaleOp, &DP_biasOp, &addOp, &actOp};
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        void* data_ptrs[] = {
            RP_YdevPtr, DP_YdevPtr, RP_scaleDevPtr, DP_scaleDevPtr, RP_biasDevPtr, DP_biasDevPtr, YdevPtr};
        int64_t uids[]   = {'y', 'a', 's', 'h', 'b', 't', 'f'};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1120: Migration of cudnnBackendExecute is not supported.
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

        return status;

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        /*
        DPCT1005:1121: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3000 || e.getCudnnStatus() == 3007)) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
            return e.getCudnnStatus();
        }
#if (CUDNN_VERSION >= 8300)
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
#endif
        return e.getCudnnStatus();
    }
}

dpct::err1
run_conv_two_global_scales(int64_t* xTensorDim,
                           int64_t* wTensorDim,
                           int64_t* yTensorDim,
                           int64_t* scaleTensorDim,
                           int convDim,
                           int64_t* conv_padA,
                           int64_t* conv_dilationA,
                           int64_t* conv_strideA,
                           void* devPtrX,
                           void* devPtrW,
                           void* devPtrScale1,
                           void* devPtrScale2,
                           void* devPtrOutput,
                           void* afterConv) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_conv_two_global_scales: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        generateStrides(xTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, xTensorDim)
                           .setStride(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dpct::library_data_t::real_half)
                           .build();

        generateStrides(scaleTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto scale1Tensor = cudnn_frontend::TensorBuilder()
                                .setDim(4, scaleTensorDim)
                                .setStride(4, stride)
                                .setId('s')
                                .setAlignment(16)
                                .setDataType(dpct::library_data_t::real_float)
                                .build();

        auto scale2Tensor = cudnn_frontend::TensorBuilder().cloneFrom(scale1Tensor, 'b').build();

        generateStrides(wTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, wTensorDim)
                           .setStride(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dpct::library_data_t::real_half)
                           .build();

        generateStrides(yTensorDim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
                                   .setDim(4, yTensorDim)
                                   .setStride(4, stride)
                                   .setId('a')  // after conv
                                   .setAlignment(16)
                                   .setDataType(dpct::library_data_t::real_half)
                                   .build();

        auto afterScale1Tensor = cudnn_frontend::TensorBuilder().cloneFrom(afterConvTensor, 'v').setVirtual().build();

        auto finalOutputTensor =
            cudnn_frontend::TensorBuilder().cloneFrom(afterConvTensor, 'y').setVirtual(false).build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << finalOutputTensor.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        // Define the scale descriptor
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                             /*
                             DPCT1007:1122: Migration of CUDNN_POINTWISE_MUL is not supported.
                             */
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setComputeType(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        std::cout << "Creating OPs " << std::endl;

        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1123: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale1_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(afterConvTensor)
                             .setbDesc(scale1Tensor)
                             .setyDesc(afterScale1Tensor)
                             .setpwDesc(scaleDesc)
                             .build();
        std::cout << scale1_op.describe() << std::endl;

        /*
        DPCT1007:1124: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale2_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(afterScale1Tensor)
                             .setbDesc(scale2Tensor)
                             .setyDesc(finalOutputTensor)
                             .setpwDesc(scaleDesc)
                             .build();
        std::cout << scale2_op.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1125: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        std::cout << conv_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is scale bias Relu conv gen_stats
        std::array<cudnn_frontend::Operation const*, 3> ops = {&conv_op, &scale1_op, &scale2_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();
        std::cout << opGraph.describe() << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

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
            } catch (cudnn_frontend::cudnnException&) {
                continue;
            }
        }
        if (plan_desc == nullptr) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3000, "run_conv_two_global_scales: No plan found to be implementing this operation graph");
        }

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        void* data_ptrs[] = {devPtrX, devPtrW, devPtrScale1, devPtrScale2, devPtrOutput, afterConv};
        int64_t uids[]    = {'x', 'w', 's', 'b', 'y', 'a'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(6, data_ptrs)
                               .setUids(6, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        dpct::err1 status =
            /*
            DPCT1007:1126: Migration of cudnnBackendExecute is not supported.
            */
            cudnnBackendExecute(handle_, plan_desc->get_backend_descriptor(), variantPack.get_raw_desc());

        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
        return status;
    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere cards
        /*
        DPCT1005:1127: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3000 || e.getCudnnStatus() == 3007)) {
            std::cout << "Fusion with float inputs is only supported on Ampere or later" << std::endl;
            return e.getCudnnStatus();
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
#if (CUDNN_VERSION >= 8300)
            CHECK(false);
#endif
            return e.getCudnnStatus();
        }
    }
}

#if (CUDNN_VERSION >= 8600)
void
run_maxpool_with_idx(int64_t* x_dim,
                     int64_t* y_dim,
                     int64_t* idx_dim,
                     void* devPtrdX,
                     void* devPtrdY,
                     void* devPtrIdx,
                     dpct::library_data_t tensorType,
                     int const nanOpt,
                     cudnn_frontend::ResampleMode_t mode,
                     cudnn_frontend::PaddingMode_t const padding_mode,
                     int32_t nbSpatialDims,
                     int64_t* windowDimA,
                     int64_t* prePaddingA,
                     int64_t* postPaddingA,
                     int64_t* strideA) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t strideTensor[4];
        generateStrides(x_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, strideTensor)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(tensorType)
                           .build();

        generateStrides(y_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto yTensor = cudnn_frontend::TensorBuilder()
                           .setDim(4, y_dim)
                           .setStrides(4, strideTensor)
                           .setId('y')  // after conv
                           .setAlignment(16)
                           .setDataType(tensorType)
                           .build();

        generateStrides(idx_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto idxTensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, idx_dim)
                             .setStrides(4, strideTensor)
                             .setId('i')
                             .setAlignment(16)
                             .setDataType(dpct::library_data_t::real_int8)
                             .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << yTensor.describe() << std::endl;
        std::cout << idxTensor.describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setNanPropagation(nanOpt)
                            .setResampleMode(mode)
                            .setPaddingMode(padding_mode)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setSpatialStride(nbSpatialDims, strideA)
                            .setPrePadding(nbSpatialDims, prePaddingA)
                            .setPostPadding(nbSpatialDims, postPaddingA)
                            .build();
        std::cout << "Initialized Pool Desc" << std::endl;
        std::cout << poolDesc.describe() << std::endl;

        // Create a maxpooling Resample Node with index tensor
        /*
        DPCT1007:1128: Migration of CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR is not supported.
        */
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setyDesc(yTensor)
                           .setidxDesc(idxTensor)
                           .setResampleDesc(poolDesc)
                           .build();
        std::cout << pool_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 1> ops = {&pool_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        // Create the variant pack and associate with the data pointers
        void* data_ptrs[] = {devPtrdX, devPtrdY, devPtrIdx};
        int64_t uids[]    = {'x', 'y', 'i'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // Trigger the execute operation
        /*
        DPCT1007:1129: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }
        /*
        DPCT1027:1130: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
        std::cout << "EXECUTE SUCCESS" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere cards
        /*
        DPCT1005:1131: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif

#if (CUDNN_VERSION >= 8600)
void
run_backward_avgpool(int64_t* dx_dim,
                     int64_t* dy_dim,
                     void* devPtrdX,
                     void* devPtrdY,
                     dpct::library_data_t tensorType,
                     int const nanOpt,
                     cudnn_frontend::ResampleMode_t mode,
                     cudnn_frontend::PaddingMode_t const padding_mode,
                     int32_t nbSpatialDims,
                     int64_t* windowDimA,
                     int64_t* prePaddingA,
                     int64_t* postPaddingA,
                     int64_t* strideA) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t strideTensor[4];
        generateStrides(dy_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto dyTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, dy_dim)
                            .setStrides(4, strideTensor)
                            .setId('y')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(tensorType)
                            .build();

        generateStrides(dx_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto dxTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, dx_dim)
                            .setStrides(4, strideTensor)
                            .setId('x')  // after conv
                            .setAlignment(16)
                            .setDataType(tensorType)
                            .build();

        std::cout << dyTensor.describe() << std::endl;
        std::cout << dxTensor.describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setNanPropagation(nanOpt)
                            .setResampleMode(mode)
                            .setPaddingMode(padding_mode)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setSpatialStride(nbSpatialDims, strideA)
                            .setPrePadding(nbSpatialDims, prePaddingA)
                            .setPostPadding(nbSpatialDims, postPaddingA)
                            .build();
        std::cout << "Initialized Pool Desc" << std::endl;
        std::cout << poolDesc.describe() << std::endl;

        // Create an average pooling Resample Node
        /*
        DPCT1007:1132: Migration of CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR is not supported.
        */
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                           .setdxDesc(dxTensor)
                           .setdyDesc(dyTensor)
                           .setResampleDesc(poolDesc)
                           .build();
        std::cout << pool_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 1> ops = {&pool_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        // Create the variant pack and associate with the data pointers
        void* data_ptrs[] = {devPtrdX, devPtrdY};
        int64_t uids[]    = {'x', 'y'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(2, data_ptrs)
                               .setUids(2, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // Trigger the execute operation
        /*
        DPCT1007:1133: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }
        /*
        DPCT1027:1134: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
        std::cout << "EXECUTE SUCCESS" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere cards
        /*
        DPCT1005:1135: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 8 && (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif

#if (CUDNN_VERSION >= 8600)
void
run_backward_maxpool(int64_t* dx_dim,
                     int64_t* dy_dim,
                     int64_t* idx_dim,
                     void* devPtrdX,
                     void* devPtrdY,
                     void* devPtrIdx,
                     dpct::library_data_t tensorType,
                     int const nanOpt,
                     cudnn_frontend::ResampleMode_t mode,
                     cudnn_frontend::PaddingMode_t const padding_mode,
                     int32_t nbSpatialDims,
                     int64_t* windowDimA,
                     int64_t* prePaddingA,
                     int64_t* postPaddingA,
                     int64_t* strideA) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        // Creates the necessary tensor descriptors
        int64_t strideTensor[4];
        generateStrides(dy_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto dyTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, dy_dim)
                            .setStrides(4, strideTensor)
                            .setId('y')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(tensorType)
                            .build();

        generateStrides(dx_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto dxTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, dx_dim)
                            .setStrides(4, strideTensor)
                            .setId('x')  // after conv
                            .setAlignment(16)
                            .setDataType(tensorType)
                            .build();

        generateStrides(idx_dim, strideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto idxTensor = cudnn_frontend::TensorBuilder()
                             .setDim(4, idx_dim)
                             .setStrides(4, strideTensor)
                             .setId('i')
                             .setAlignment(16)
                             .setDataType(dpct::library_data_t::real_int8)
                             .build();

        std::cout << dyTensor.describe() << std::endl;
        std::cout << dxTensor.describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder_v8()
                            .setComputeType(dpct::library_data_t::real_float)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setNanPropagation(nanOpt)
                            .setResampleMode(mode)
                            .setPaddingMode(padding_mode)
                            .setSpatialDim(nbSpatialDims, windowDimA)
                            .setSpatialStride(nbSpatialDims, strideA)
                            .setPrePadding(nbSpatialDims, prePaddingA)
                            .setPostPadding(nbSpatialDims, postPaddingA)
                            .build();
        std::cout << "Initialized Pool Desc" << std::endl;
        std::cout << poolDesc.describe() << std::endl;

        // Create a maxpooling Resample Node with index tensor
        /*
        DPCT1007:1136: Migration of CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR is not supported.
        */
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                           .setdxDesc(dxTensor)
                           .setdyDesc(dyTensor)
                           .setidxDesc(idxTensor)
                           .setResampleDesc(poolDesc)
                           .build();
        std::cout << pool_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution bias scale activation
        std::array<cudnn_frontend::Operation const*, 1> ops = {&pool_op};
        auto opGraph                                        = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        // Create the variant pack and associate with the data pointers
        void* data_ptrs[] = {devPtrdX, devPtrdY, devPtrIdx};
        int64_t uids[]    = {'x', 'y', 'i'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(3, data_ptrs)
                               .setUids(3, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;

        // Trigger the execute operation
        /*
        DPCT1007:1137: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }
        /*
        DPCT1027:1138: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);
        cudnn_frontend::throw_if(
            [dpct::err1]() {
                                                  return (status != 0);
            },
            "Plan execute error",
            status);
        std::cout << "EXECUTE SUCCESS" << std::endl;

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere cards
        /*
        DPCT1005:1139: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() != 8 && (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for Ampere GPUs" << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif

#if (CUDNN_VERSION >= 8400)
void
run_bn_bwd_weight(int64_t* xDim,
                  int64_t* dyDim,
                  int64_t* wDim,
                  int64_t* scaleDim,
                  void* x_bn_fwd,
                  void* w_fwd,
                  void* dy,
                  void* dy_bn,
                  void* mean,
                  void* inv_var,
                  void* scale,
                  void* bias,
                  void* d_scale,
                  void* d_bias,
                  void* eqscale_dy,
                  void* eqscale_x,
                  void* eqbias) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        if (check_device_arch_newer_than("ampere") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_conv_scale_bias_relu_gen_index_selection: Sample requires Ampere or above GPU");
        }

        dpct::library_data_t computeType = dpct::library_data_t::real_float;

        // Creates the necessary tensor descriptors
        int64_t xstrideTensor[4];
        int64_t dystrideTensor[4];
        int64_t wstrideTensor[4];
        generateStrides(xDim, xstrideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        generateStrides(dyDim, dystrideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);
        generateStrides(wDim, wstrideTensor, 4, dpct::dnnl::memory_format_tag::nhwc);

        int64_t perChannelStride[4];
        generateStrides(scaleDim, perChannelStride, 4, dpct::dnnl::memory_format_tag::nhwc);

        auto tensor_create = [](int64_t* stride, int64_t* dim, dpct::library_data_t type, int64_t id, bool is_virtual) {
            return cudnn_frontend::TensorBuilder()
                .setDim(4, dim)
                .setStride(4, stride)
                .setId(id)
                .setAlignment(16)
                .setDataType(type)
                .setVirtual(is_virtual)
                .build();
        };

        auto pointwise_create = [](cudnnPointwiseMode_t mode) {
            return cudnn_frontend::PointWiseDescBuilder()
                .setMode(mode)
                .setComputeType(dpct::library_data_t::real_float)
                .build();
        };

        auto pointwise_op_create = [](cudnn_frontend::Tensor& x,
                                      cudnn_frontend::Tensor& s,
                                      cudnn_frontend::Tensor& y,
                                      cudnn_frontend::PointWiseDesc& pw) {
            /*
            DPCT1007:1140: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
            */
            return cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(x)
                .setbDesc(s)
                .setyDesc(y)
                .setpwDesc(pw)
                .build();
        };

        auto x_tensor_bn_fwd = tensor_create(xstrideTensor, xDim, dpct::library_data_t::real_half, 100, false);
        auto w_tensor        = tensor_create(wstrideTensor, wDim, dpct::library_data_t::real_half, 101, false);
        auto dy_tensor       = tensor_create(dystrideTensor, dyDim, dpct::library_data_t::real_half, 102, false);
        auto dy_bn_tensor    = tensor_create(xstrideTensor, xDim, dpct::library_data_t::real_half, 103, false);

        auto scaleTensor  = tensor_create(perChannelStride, scaleDim, computeType, 200, false);
        auto biasTensor   = tensor_create(perChannelStride, scaleDim, computeType, 201, false);
        auto meanTensor   = tensor_create(perChannelStride, scaleDim, computeType, 202, false);
        auto invVarTensor = tensor_create(perChannelStride, scaleDim, computeType, 203, false);

        auto d_scaleTensor    = tensor_create(perChannelStride, scaleDim, computeType, 300, false);
        auto d_biasTensor     = tensor_create(perChannelStride, scaleDim, computeType, 301, false);
        auto eqscale_dyTensor = tensor_create(perChannelStride, scaleDim, computeType, 302, false);
        auto eqscale_xTensor  = tensor_create(perChannelStride, scaleDim, computeType, 303, false);
        auto eqbiasTensor     = tensor_create(perChannelStride, scaleDim, computeType, 304, false);

        auto after_scaleTensor  = tensor_create(xstrideTensor, xDim, computeType, 400, true);
        auto after_biasTensor   = tensor_create(xstrideTensor, xDim, computeType, 401, true);
        auto after_meanTensor   = tensor_create(xstrideTensor, xDim, computeType, 402, true);
        auto after_invVarTensor = tensor_create(xstrideTensor, xDim, computeType, 403, true);

        auto after_dgrad_tensor = tensor_create(xstrideTensor, xDim, dpct::library_data_t::real_half, 500, true);

        // Define the pointwise descriptor
        /*
        DPCT1007:1141: Migration of CUDNN_POINTWISE_MUL is not supported.
        */
        auto scaleDesc = pointwise_create(CUDNN_POINTWISE_MUL);
        /*
        DPCT1007:1142: Migration of CUDNN_POINTWISE_ADD is not supported.
        */
        auto biasDesc = pointwise_create(CUDNN_POINTWISE_ADD);
        /*
        DPCT1007:1143: Migration of CUDNN_POINTWISE_ADD is not supported.
        */
        auto addDesc = pointwise_create(CUDNN_POINTWISE_ADD);
        /*
        DPCT1007:1144: Migration of CUDNN_POINTWISE_MUL is not supported.
        */
        auto mulDesc = pointwise_create(CUDNN_POINTWISE_MUL);
        /*
        DPCT1007:1145: Migration of CUDNN_POINTWISE_RELU_BWD is not supported.
        */
        auto bwdReluDesc = pointwise_create(CUDNN_POINTWISE_RELU_BWD);

        // Create Pointwise Operations
        auto addOpDesc     = pointwise_op_create(x_tensor_bn_fwd, meanTensor, after_meanTensor, addDesc);
        auto mulOpDesc     = pointwise_op_create(after_meanTensor, invVarTensor, after_invVarTensor, mulDesc);
        auto scaleOpDesc   = pointwise_op_create(after_invVarTensor, scaleTensor, after_scaleTensor, scaleDesc);
        auto biasOpDesc    = pointwise_op_create(after_scaleTensor, biasTensor, after_biasTensor, biasDesc);
        /*
        DPCT1007:1146: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto bwdReluOpDesc = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                 .setdyDesc(after_dgrad_tensor)
                                 .setxDesc(after_biasTensor)
                                 .setdxDesc(dy_bn_tensor)
                                 .setpwDesc(bwdReluDesc)
                                 .build();

        // Create dgrad desc and operation
        int64_t convDim      = 2;
        int64_t padding[]    = {1, 1};
        int64_t dilation[]   = {1, 1};
        int64_t convstride[] = {1, 1};

        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setComputeType(computeType)
                            .setMathMode(1)
                            .setSpatialDimCount(convDim)
                            .setSpatialStride(convDim, convstride)
                            .setPrePadding(convDim, padding)
                            .setPostPadding(convDim, padding)
                            .setDilation(convDim, dilation)
                            .build();

        /*
        DPCT1007:1147: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR is not supported.
        */
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                           .setdyDesc(dy_tensor)
                           .setwDesc(w_tensor)
                           .setdxDesc(after_dgrad_tensor)
                           .setcDesc(convDesc)
                           .setAlpha(1.0f)
                           .setBeta(0.0f)
                           .build();

        /*
        DPCT1007:1148: Migration of CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR is not supported.
        */
        auto bn_bwd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
                             .setComputeType(computeType)
                             .setxDesc(x_tensor_bn_fwd)
                             .setSavedMeanAndInvVar(meanTensor, invVarTensor)
                             .setScale(scaleTensor)
                             .setdyDesc(dy_bn_tensor)
                             .setEqScalesAndBias(eqscale_dyTensor, eqscale_xTensor, eqbiasTensor)
                             .setDScaleAndDBias(d_scaleTensor, d_biasTensor)
                             .build();

        // Create an Operation Graph. In this case it is convolution scale bias add activation
        std::array<cudnn_frontend::Operation const*, 7> ops = {
            &conv_op, &addOpDesc, &mulOpDesc, &scaleOpDesc, &biasOpDesc, &bwdReluOpDesc, &bn_bwd_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        // Create engine configuration
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<2>(
            {"heuristics_instant", "heuristics_fallback"}, opGraph, ::allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }

        void* data_ptrs[] = {
            x_bn_fwd, w_fwd, dy, dy_bn, scale, bias, mean, inv_var, d_scale, d_bias, eqscale_dy, eqscale_x, eqbias};
        int64_t uids[]   = {100, 101, 102, 103, 200, 201, 202, 203, 300, 301, 302, 303, 304};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(13, data_ptrs)
                               .setUids(13, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1149: Migration of cudnnBackendExecute is not supported.
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
        /*
        DPCT1027:1150: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

    } catch (cudnn_frontend::cudnnException& e) {
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));

        // this example is only for Ampere and Hopper cards
        bool is_supported_on_ampere = is_ampere_arch();
        bool is_supported_on_hopper = is_hopper_arch() && (dpct::dnnl::get_version() >= 8900);
        if (((!is_supported_on_hopper) && (!is_supported_on_ampere)) &&
            (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            SKIP("Example is only supported for Ampere and Hopper GPUs");
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif
