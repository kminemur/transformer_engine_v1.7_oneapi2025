#include <dpct/dnnl_utils.hpp>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "fp8_sample.h"
#include <cudnn_frontend.h>
#include "../utils/error_util.h"

using namespace cudnn_frontend;

ExecutionPlan_v8
get_exec_plan_from_heuristics(OperationGraph_v8&& opGraph, dpct::dnnl::engine_ext handle) {
    /*
    DPCT1007:1172: Migration of CUDNN_HEUR_MODE_INSTANT is not supported.
    */
    auto heuristics = EngineHeuristicsBuilder().setOperationGraph(opGraph).setHeurMode(CUDNN_HEUR_MODE_INSTANT).build();

    auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    auto plan_builder = [&]() -> ExecutionPlan {
        for (auto& ecfg : engine_config) {
            try {
                auto plan = ExecutionPlanBuilder()
                                .setHandle(dpct::dnnl::engine_ext)
                                .setEngineConfig(ecfg, opGraph.getTag())
                                .build();
                return plan;
            } catch (cudnnException& e) {
                continue;
            }
        }
        return ExecutionPlanBuilder().setHandle(handle).setEngineConfig(engine_config[0], opGraph.getTag()).build();
    };

    return plan_builder();
}

#if (CUDNN_VERSION >= 8600)
void
run_fp8_conv_scale(int64_t* x_dim,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   int64_t* scale_dim,
                   dpct::library_data_t dataType,
                   int convDim,
                   int64_t* conv_padA,
                   int64_t* conv_dilationA,
                   int64_t* conv_strideA,
                   void* devPtrX,
                   void* devPtrW,
                   void* devPtrY,
                   void* devPtrScale) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_fp8_conv_scale: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        ::generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        ::generateStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        ::generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('y')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dpct::library_data_t::real_float)
                                   .build();

        auto afterScaleTensor =
            TensorBuilder().cloneFrom(afterConvTensor, 'a').setVirtual(false).setDataType(dataType).build();

        ::generateStrides(scale_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto scaleTensor = TensorBuilder()
                               .setDim(4, scale_dim)
                               .setStrides(4, stride)
                               .setId('s')  // after conv
                               .setAlignment(16)
                               .setDataType(dpct::library_data_t::real_float)
                               .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << scaleTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;

        // Define the scale descriptor
        /*
        DPCT1007:1173: Migration of CUDNN_POINTWISE_MUL is not supported.
        */
        auto scaleDesc = PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = ConvDescBuilder()
                            .setDataType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1174: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
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
        DPCT1007:1175: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(conv_op.getOutputTensor())
                            .setbDesc(scaleTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution scale
        std::array<Operation const*, 2> ops = {&conv_op, &scale_op};

        auto opGraph = OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();

        auto plan = get_exec_plan_from_heuristics(std::move(opGraph), handle_);
        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrY, devPtrScale};
        int64_t uids[]    = {'x', 'w', 'a', 's'};
        auto variantPack  = VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1176: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1177: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        throw_if(
            [dpct::err1]() {
                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnnException& e) {
        // this example is only for Hopper cards
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        /*
        DPCT1005:1178: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 9 && (e.getCudnnStatus() == 3000 || e.getCudnnStatus() == 3007)) {
            std::cout << "Fusion with fp8 inputs is only supported on Hopper or later" << std::endl;
            return;
        }

        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_fp8_conv_descale_descale_amax_scale(int64_t* x_dim,
                                        int64_t* w_dim,
                                        int64_t* y_dim,
                                        int64_t* r_dim,
                                        int64_t* scale_dim,
                                        dpct::library_data_t dataType,
                                        int convDim,
                                        int64_t* conv_padA,
                                        int64_t* conv_dilationA,
                                        int64_t* conv_strideA,
                                        void* devPtrX,
                                        void* devPtrW,
                                        void* devPtrR,
                                        void* devPtrOutput,
                                        void* devPtrDescale1,
                                        void* devPtrDescale2,
                                        void* devPtrScale) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_fp8_conv_descale_descale_amax_scale: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];
        ::generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dataType)
                           .build();
        ::generateStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        ::generateStrides(r_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto amaxTensor = TensorBuilder()
                              .setDim(4, r_dim)
                              .setStrides(4, stride)
                              .setId('r')  // output
                              .setAlignment(16)
                              .setDataType(dpct::library_data_t::real_float)
                              .build();

        ::generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvTensor = TensorBuilder()
                                   .setDim(4, y_dim)
                                   .setStrides(4, stride)
                                   .setId('y')  // after conv
                                   .setAlignment(16)
                                   .setVirtual()
                                   .setDataType(dpct::library_data_t::real_float)
                                   .build();

        auto afterDescale1Tensor = TensorBuilder().cloneFrom(afterConvTensor, 'a').build();

        auto afterDescale2Tensor = TensorBuilder().cloneFrom(afterConvTensor, 'b').build();

        auto fp8OutputTensor =
            TensorBuilder().cloneFrom(afterConvTensor, 'c').setVirtual(false).setDataType(dataType).build();

        ::generateStrides(scale_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto descaleTensor1 = TensorBuilder()
                                  .setDim(4, scale_dim)
                                  .setStrides(4, stride)
                                  .setId('s')
                                  .setAlignment(16)
                                  .setDataType(dpct::library_data_t::real_float)
                                  .build();

        auto descaleTensor2 = TensorBuilder().cloneFrom(descaleTensor1, 't').build();

        auto scaleTensor = TensorBuilder().cloneFrom(descaleTensor1, 'u').build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << scaleTensor.describe() << std::endl;
        std::cout << afterConvTensor.describe() << std::endl;

        // Define the scale descriptor
        /*
        DPCT1007:1179: Migration of CUDNN_POINTWISE_MUL is not supported.
        */
        auto scaleDesc = PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the reduction descriptor
        auto redunctionDesc = ReductionDescBuilder()
                                  .setMathPrecision(dpct::library_data_t::real_float)
                                  .setReductionOp(dpct::dnnl::reduction_op::amax)
                                  .build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = ConvDescBuilder()
                            .setDataType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1180: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        auto conv_op = OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
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
        DPCT1007:1181: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto descale_op1 = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(afterConvTensor)
                               .setbDesc(descaleTensor1)
                               .setyDesc(afterDescale1Tensor)
                               .setpwDesc(scaleDesc)
                               .build();
        std::cout << descale_op1.describe() << std::endl;

        /*
        DPCT1007:1182: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto descale_op2 = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(afterDescale1Tensor)
                               .setbDesc(descaleTensor2)
                               .setyDesc(afterDescale2Tensor)
                               .setpwDesc(scaleDesc)
                               .build();
        std::cout << descale_op2.describe() << std::endl;

        /*
        DPCT1007:1183: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(afterDescale2Tensor)
                            .setbDesc(scaleTensor)
                            .setyDesc(fp8OutputTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a reduction add Node.
        /*
        DPCT1007:1184: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
        */
        auto reduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterDescale2Tensor)
                                .setyDesc(amaxTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution descale descale amax scale
        std::array<Operation const*, 5> ops = {&conv_op, &descale_op1, &descale_op2, &scale_op, &reduction_op};

        auto opGraph = OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();

        auto plan = get_exec_plan_from_heuristics(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrR, devPtrDescale1, devPtrDescale2, devPtrScale, devPtrOutput};
        int64_t uids[]    = {'x', 'w', 'r', 's', 't', 'u', 'c'};
        auto variantPack  = VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1185: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudaErr(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1186: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        throw_if(
            [dpct::err1]() {
                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnnException& e) {
        // this example is only for Hopper cards
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        /*
        DPCT1005:1187: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 9 && (e.getCudnnStatus() == 3000 || e.getCudnnStatus() == 3007)) {
            std::cout << "Fusion with fp8 inputs is only supported on Hopper or later" << std::endl;
            return;
        }

        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_tranpose_scale_convert_fp16_fp8_amax(int64_t* x_dim,
                                         int64_t* y_dim,
                                         int64_t* r_dim,
                                         int64_t* scale_dim,
                                         dpct::library_data_t dataType,
                                         void* devPtrX,
                                         void* devPtrR,
                                         void* devPtrOutput,
                                         void* devPtrScale) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_tranpose_scale_convert_fp16_fp8_amax: Sample requires Ampere or above GPU");
        }

        // Creates the necessary tensor descriptors
        int64_t stride[4];
        ::generateStrides(x_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto xTensor = TensorBuilder()
                           .setDim(4, x_dim)
                           .setStrides(4, stride)
                           .setId('x')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(dpct::library_data_t::real_half)  // Half as input
                           .build();

        ::generateStrides(scale_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto scaleTensor = TensorBuilder()
                               .setDim(4, scale_dim)
                               .setStrides(4, stride)
                               .setId('s')
                               .setAlignment(16)
                               .setDataType(dpct::library_data_t::real_float)
                               .build();

        ::generateStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterScaleTensor = TensorBuilder()
                                    .setDim(4, y_dim)
                                    .setStrides(4, stride)
                                    .setId('a')  // after transpose + convert
                                    .setAlignment(16)
                                    .setDataType(dpct::library_data_t::real_float)  // Transpose + convert to FP8
                                    .setVirtual()
                                    .build();

        // Tranposed from NWHC to CHWN
        ::generate4dTransposeStrides(y_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto afterConvertTensor = TensorBuilder()
                                      .setDim(4, y_dim)
                                      .setStrides(4, stride)
                                      .setId('y')  // after transpose + convert
                                      .setAlignment(16)
                                      .setDataType(dataType)  // Transpose + convert to FP8
                                      .build();

        ::generateStrides(r_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto amaxTensor = TensorBuilder()
                              .setDim(4, r_dim)
                              .setStrides(4, stride)
                              .setId('r')  // output
                              .setAlignment(16)
                              .setDataType(dpct::library_data_t::real_float)
                              .build();

        std::cout << xTensor.describe() << std::endl;
        std::cout << scaleTensor.describe() << std::endl;
        std::cout << afterConvertTensor.describe() << std::endl;

        // Define the scale descriptor
        /*
        DPCT1007:1188: Migration of CUDNN_POINTWISE_MUL is not supported.
        */
        auto scaleDesc = PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the convert descriptor
        auto identityDesc =
            /*
            DPCT1007:1189: Migration of CUDNN_POINTWISE_IDENTITY is not supported.
            */
            PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_IDENTITY)
                .setMathPrecision(dpct::library_data_t::real_float)
                .build();
        std::cout << identityDesc.describe() << std::endl;

        // Define the reduction descriptor
        auto redunctionDesc = ReductionDescBuilder()
                                  .setMathPrecision(dpct::library_data_t::real_float)
                                  .setReductionOp(dpct::dnnl::reduction_op::amax)
                                  .build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1190: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(xTensor)
                            .setbDesc(scaleTensor)
                            .setyDesc(afterScaleTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create transpose + convert node
        /*
        DPCT1007:1191: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto convert_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                              .setxDesc(afterScaleTensor)
                              .setyDesc(afterConvertTensor)
                              .setpwDesc(identityDesc)
                              .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a reduction add Node.
        /*
        DPCT1007:1192: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
        */
        auto reduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(xTensor)
                                .setyDesc(amaxTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is scale transpose amax
        std::array<Operation const*, 3> ops = {&scale_op, &convert_op, &reduction_op};

        auto opGraph = OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();

        auto plan = get_exec_plan_from_heuristics(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrX, devPtrR, devPtrScale, devPtrOutput};
        int64_t uids[]    = {'x', 'r', 's', 'y'};
        auto variantPack  = VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1193: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        checkCudaErr(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        /*
        DPCT1027:1194: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        throw_if(
            [dpct::err1]() {
                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnnException& e) {
        // this example is only for Hopper cards
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        /*
        DPCT1005:1195: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 9 && (e.getCudnnStatus() == 3000 || e.getCudnnStatus() == 3007)) {
            std::cout << "Fusion with fp8 inputs is only supported on Hopper or later" << std::endl;
            return;
        }
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}

void
run_fp8_dgrad_descale_descale_amax_scale(int64_t* dx_dim,
                                         int64_t* w_dim,
                                         int64_t* dy_dim,
                                         int64_t* r_dim,
                                         int64_t* scale_dim,
                                         dpct::library_data_t dataType,
                                         int convDim,
                                         int64_t* conv_padA,
                                         int64_t* conv_dilationA,
                                         int64_t* conv_strideA,
                                         void* devPtrdX,
                                         void* devPtrW,
                                         void* devPtrR,
                                         void* devPtrdY,
                                         void* devPtrDescale1,
                                         void* devPtrDescale2,
                                         void* devPtrScale) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));
        if (check_device_arch_newer_than("hopper") == false) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3007, "run_fp8_dgrad_descale_descale_amax_scale: Sample requires Ampere or above GPU");
        }
        // Creates the necessary tensor descriptors
        int64_t stride[4];

        ::generateStrides(dy_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto dyTensor = TensorBuilder()
                            .setDim(4, dy_dim)
                            .setStrides(4, stride)
                            .setId('y')  // after conv
                            .setAlignment(16)
                            .setDataType(dataType)
                            .build();

        ::generate4dTransposeStrides(w_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto wTensor = TensorBuilder()
                           .setDim(4, w_dim)
                           .setStrides(4, stride)
                           .setId('w')
                           .setAlignment(16)
                           .setDataType(dataType)
                           .build();

        ::generateStrides(dx_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto dxTensor = TensorBuilder()
                            .setDim(4, dx_dim)
                            .setStrides(4, stride)
                            .setId('x')
                            .setVirtual()      // after dgrad is virtual
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(dpct::library_data_t::real_float)
                            .build();

        ::generateStrides(scale_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto dyDescaleTensor = TensorBuilder()
                                   .setDim(4, scale_dim)
                                   .setStrides(4, stride)
                                   .setId('s')
                                   .setAlignment(16)
                                   .setDataType(dpct::library_data_t::real_float)
                                   .build();

        auto afterDescale1Tensor = TensorBuilder().cloneFrom(dxTensor, 'a').build();

        auto wDescaleTensor = TensorBuilder().cloneFrom(dyDescaleTensor, 't').build();

        auto afterDescale2Tensor = TensorBuilder().cloneFrom(dxTensor, 'b').build();

        auto dxScaleTensor = TensorBuilder().cloneFrom(dyDescaleTensor, 'u').build();

        auto fp8OutputTensor = TensorBuilder().cloneFrom(dxTensor, 'c').setVirtual(false).setDataType(dataType).build();

        ::generateStrides(r_dim, stride, 4, dpct::dnnl::memory_format_tag::nhwc);
        auto amaxTensor = TensorBuilder()
                              .setDim(4, r_dim)
                              .setStrides(4, stride)
                              .setId('r')  // output
                              .setAlignment(16)
                              .setDataType(dpct::library_data_t::real_float)
                              .build();

        std::cout << dxTensor.describe() << std::endl;
        std::cout << wTensor.describe() << std::endl;
        std::cout << dxScaleTensor.describe() << std::endl;
        std::cout << dyTensor.describe() << std::endl;

        // Define the scale descriptor
        /*
        DPCT1007:1196: Migration of CUDNN_POINTWISE_MUL is not supported.
        */
        auto scaleDesc = PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_MUL)
                             .setMathPrecision(dpct::library_data_t::real_float)
                             .build();
        std::cout << scaleDesc.describe() << std::endl;

        // Define the reduction descriptor
        auto redunctionDesc = ReductionDescBuilder()
                                  .setMathPrecision(dpct::library_data_t::real_float)
                                  .setReductionOp(dpct::dnnl::reduction_op::amax)
                                  .build();
        std::cout << redunctionDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = ConvDescBuilder()
                            .setDataType(dpct::library_data_t::real_float)
                            .setMathMode(1)
                            .setNDims(convDim)
                            .setStrides(convDim, conv_strideA)
                            .setPrePadding(convDim, conv_padA)
                            .setPostPadding(convDim, conv_padA)
                            .setDilation(convDim, conv_dilationA)
                            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float beta  = 0.0f;

        // Create a convolution Node
        /*
        DPCT1007:1197: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR is not supported.
        */
        auto dgrad_op = OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                            .setdyDesc(dyTensor)
                            .setwDesc(wTensor)
                            .setdxDesc(dxTensor)
                            .setcDesc(convDesc)
                            .setAlpha(alpha)
                            .setBeta(beta)
                            .build();
        std::cout << dgrad_op.describe() << std::endl;

        // Create a Multiplication Node with scaling parameters.
        /*
        DPCT1007:1198: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto descale_op1 = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(dxTensor)
                               .setbDesc(dyDescaleTensor)
                               .setyDesc(afterDescale1Tensor)
                               .setpwDesc(scaleDesc)
                               .build();
        std::cout << descale_op1.describe() << std::endl;

        /*
        DPCT1007:1199: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto descale_op2 = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                               .setxDesc(afterDescale1Tensor)
                               .setbDesc(wDescaleTensor)
                               .setyDesc(afterDescale2Tensor)
                               .setpwDesc(scaleDesc)
                               .build();
        std::cout << descale_op2.describe() << std::endl;

        /*
        DPCT1007:1200: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        auto scale_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                            .setxDesc(afterDescale2Tensor)
                            .setbDesc(dxScaleTensor)
                            .setyDesc(fp8OutputTensor)
                            .setpwDesc(scaleDesc)
                            .build();
        std::cout << scale_op.describe() << std::endl;

        // Create a reduction add Node.
        /*
        DPCT1007:1201: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
        */
        auto reduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterDescale2Tensor)
                                .setyDesc(amaxTensor)
                                .setreductionDesc(redunctionDesc)
                                .build();
        std::cout << reduction_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is dgrad descale descale amax scale
        std::array<Operation const*, 5> ops = {&dgrad_op, &descale_op1, &descale_op2, &scale_op, &reduction_op};

        auto opGraph = OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();

        auto plan = get_exec_plan_from_heuristics(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device((size_t)workspace_size, q_ct1)));
        }
        void* data_ptrs[] = {devPtrdX, devPtrW, devPtrR, devPtrDescale1, devPtrDescale2, devPtrScale, devPtrdY};
        int64_t uids[]    = {'c', 'w', 'r', 's', 't', 'u', 'y'};
        auto variantPack  = VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(7, data_ptrs)
                               .setUids(7, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1202: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1203: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
        */
        checkCudnnErr(0);

        throw_if(
            [dpct::err1]() {
                                  return (status != 0);
            },
            "Plan execute error",
            status);

    } catch (cudnnException& e) {
        // this example is only for Hopper cards
        dpct::device_info prop;
        checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(prop, dpct::dev_mgr::instance().get_device(0))));
        /*
        DPCT1005:1204: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (prop.get_major_version() < 9 && (e.getCudnnStatus() == 3000 || e.getCudnnStatus() == 3007)) {
            std::cout << "Fusion with fp8 inputs is only supported on Hopper or later" << std::endl;
            return;
        }

        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        CHECK(false);
    }
}
#endif
