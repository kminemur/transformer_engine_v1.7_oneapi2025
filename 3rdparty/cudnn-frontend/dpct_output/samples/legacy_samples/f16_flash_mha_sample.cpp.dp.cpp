#include <dpct/dnnl_utils.hpp>
/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include "f16_flash_mha_sample.h"
#include <cudnn_frontend.h>
#include "../utils/error_util.h"

#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define D_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define D_SEED_ID 17
#define D_OFFSET_ID 18
#define S_STATS_ID 19
#define S_SUM_ID 20
#define SCALE_PROB 21
#define K_TRANSPOSE_ID 22
#define dQ_ACCUM_ID 23

#define VIRTUAL_ID 30

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

#if (CUDNN_VERSION >= 8900)
static cudnn_frontend::Tensor
tensor_create(dpct::library_data_t type,
              int64_t id,
              int64_t const* dim,
              int64_t const* stride,
              bool is_virtual,
              bool is_value) {
    int nbDims          = 4;
    auto tensor_created = cudnn_frontend::TensorBuilder()
                              .setDim(nbDims, dim)
                              .setStride(nbDims, stride)
                              .setId(id)
                              .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                              .setDataType(type)
                              .setVirtual(is_virtual)
                              .setByValue(is_value)
                              .build();
    std::cout << tensor_created.describe() << std::endl;
    return tensor_created;
}

static cudnn_frontend::PointWiseDesc
pw_desc_create(dpct::library_data_t type, cudnnPointwiseMode_t mode) {
    auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder().setMode(mode).setComputeType(type).build();

    std::cout << pw_desc_created.describe() << std::endl;
    return pw_desc_created;
}

static cudnn_frontend::Operation
unary_pw_op_create(cudnn_frontend::Tensor const& xDesc,
                   cudnn_frontend::Tensor const& yDesc,
                   cudnn_frontend::PointWiseDesc const& pwDesc) {
    /*
    DPCT1007:1264: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
    */
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(xDesc)
                             .setyDesc(yDesc)
                             .setpwDesc(pwDesc)
                             .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Operation
binary_pw_op_create(cudnn_frontend::Tensor const& xDesc,
                    cudnn_frontend::Tensor const& bDesc,
                    cudnn_frontend::Tensor const& yDesc,
                    cudnn_frontend::PointWiseDesc const& pwDesc) {
    /*
    DPCT1007:1265: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
    */
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(xDesc)
                             .setbDesc(bDesc)
                             .setyDesc(yDesc)
                             .setpwDesc(pwDesc)
                             .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Operation
ternary_pw_op_create(cudnn_frontend::Tensor const& xDesc,
                     cudnn_frontend::Tensor const& bDesc,
                     cudnn_frontend::Tensor const& tDesc,
                     cudnn_frontend::Tensor const& yDesc,
                     cudnn_frontend::PointWiseDesc const& pwDesc) {
    /*
    DPCT1007:1266: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
    */
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                             .setxDesc(xDesc)
                             .setbDesc(bDesc)
                             .settDesc(tDesc)
                             .setyDesc(yDesc)
                             .setpwDesc(pwDesc)
                             .build();
    std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Tensor
createScale(int64_t b,
            int64_t h,
            int64_t s_q,
            int64_t s_kv,
            int64_t d,
            MHA_Layout layout,
            dpct::library_data_t tensorType,
            const cudnn_frontend::Tensor& sTensor,
            std::vector<cudnn_frontend::Operation>& ops) {
    // scale
    int64_t scale_dim[4]    = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    int64_t s_dim[4] = {b, h, s_q, s_kv};
    int64_t s_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    auto scaleTensor  = tensor_create(tensorType, S_CONST_ID, scale_dim, scale_stride, false, true);  // is by value
    auto sScaleTensor = tensor_create(tensorType, VIRTUAL_ID + 2000, s_dim, s_stride, true, false);   // is virtual

    // Define the scale descriptor
    /*
    DPCT1007:1267: Migration of CUDNN_POINTWISE_MUL is not supported.
    */
    auto scaleDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);

    // Create a Scale Node.
    auto scale_op = binary_pw_op_create(sTensor, scaleTensor, sScaleTensor, scaleDesc);

    ops.push_back(std::move(scale_op));
    return sScaleTensor;
}

static cudnn_frontend::Tensor
createQKBMM(int64_t b,
            int64_t h,
            int64_t s_q,
            int64_t s_kv,
            int64_t d,
            MHA_Layout layout,
            dpct::library_data_t tensorType,
            std::vector<cudnn_frontend::Operation>& ops) {
    // Creates the necessary tensor descriptors
    int64_t q_dim[4] = {b, h, s_q, d};
    int64_t q_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

    int64_t k_dim[4] = {b, h, d, s_kv};
    int64_t k_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix_Transpose);

    int64_t s_dim[4] = {b, h, s_q, s_kv};
    int64_t s_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    auto qTensor          = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
    auto kTransposeTensor = tensor_create(tensorType, K_ID, k_dim, k_stride, false, false);  // is virtual
    // first GEMM output
    auto sTensor =
        tensor_create(dpct::library_data_t::real_float, VIRTUAL_ID + 1, s_dim, s_stride, true, false);  // is virtual

    // Define the matmul 1 desc
    auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
    std::cout << matmul_1_Desc.describe() << std::endl;

    // Create a matmul 1 Node
    /*
    DPCT1007:1268: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
    */
    auto matmul_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                          .setaMatDesc(qTensor)
                          .setbMatDesc(kTransposeTensor)
                          .setcMatDesc(sTensor)
                          .setmatmulDesc(matmul_1_Desc)
                          .build();

    std::cout << matmul_op1.describe() << std::endl;

    ops.push_back(std::move(matmul_op1));

    return sTensor;
}

static cudnn_frontend::Tensor
createCausalMask(int64_t b,
                 int64_t h,
                 int64_t s_q,
                 int64_t s_kv,
                 int64_t d,
                 MHA_Layout layout,
                 dpct::library_data_t tensorType,
                 std::vector<cudnn_frontend::Operation>& ops,
                 cudnn_frontend::Tensor& prevBlockOutputTensor) {
    CUDNN_FRONTEND_UNUSED(d);
    CUDNN_FRONTEND_UNUSED(layout);
    CUDNN_FRONTEND_UNUSED(tensorType);

    cudnn_frontend::throw_if(ops.size() == 0, "Padding Mask constructed incorrectly as the first one", 2000);

    // subtraction output
    int64_t afterBMM1_dim[4]    = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t maskVal_dim[4]    = {1, 1, 1, 1};
    int64_t maskVal_stride[4] = {1, 1, 1, 1};

    // mask value to put in the masked pixels
    auto maskValTensor = tensor_create(
        dpct::library_data_t::real_float, MASK_VAL_ID, maskVal_dim, maskVal_stride, false, true);  // is by value
    // gen index row output
    auto rowIndexTensor = tensor_create(dpct::library_data_t::real_float,
                                        VIRTUAL_ID + 100,
                                        afterBMM1_dim,
                                        afterBMM1_stride,
                                        true,
                                        false);  // is virtual
    // gen index column output
    auto columnIndexTensor = tensor_create(dpct::library_data_t::real_float,
                                           VIRTUAL_ID + 101,
                                           afterBMM1_dim,
                                           afterBMM1_stride,
                                           true,
                                           false);  // is virtual
    // create causal mask (row >= col)
    auto causalMaskTensor = tensor_create(
        /*
        DPCT1007:1269: Migration of CUDNN_DATA_BOOLEAN is not supported.
        */
        CUDNN_DATA_BOOLEAN,
        VIRTUAL_ID + 106,
        afterBMM1_dim,
        afterBMM1_stride,
        true,
        false);  // is virtual

    // output after masking
    auto maskOutputTensor = tensor_create(dpct::library_data_t::real_float,
                                          VIRTUAL_ID + 107,
                                          afterBMM1_dim,
                                          afterBMM1_stride,
                                          true,
                                          false);  // is virtual

    // Define the gen index for row descriptor
    auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                               /*
                               DPCT1007:1270: Migration of CUDNN_POINTWISE_GEN_INDEX is not supported.
                               */
                               .setMode(CUDNN_POINTWISE_GEN_INDEX)
                               .setAxis(2)
                               .setComputeType(dpct::library_data_t::real_float)
                               .build();
    std::cout << genIndexRowDesc.describe() << std::endl;

    // Create a gen index Node.
    auto genIndexRow_op = unary_pw_op_create(prevBlockOutputTensor, rowIndexTensor, genIndexRowDesc);
    std::cout << genIndexRow_op.describe() << std::endl;

    // Define the gen index for row descriptor
    auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                                  /*
                                  DPCT1007:1271: Migration of CUDNN_POINTWISE_GEN_INDEX is not supported.
                                  */
                                  .setMode(CUDNN_POINTWISE_GEN_INDEX)
                                  .setAxis(3)
                                  .setComputeType(dpct::library_data_t::real_float)
                                  .build();
    std::cout << genIndexColumnDesc.describe() << std::endl;

    // Create a gen index Node.
    auto genIndexColumn_op = unary_pw_op_create(prevBlockOutputTensor, columnIndexTensor, genIndexColumnDesc);

    // Define the greater than equal to comparison descriptor
    /*
    DPCT1007:1272: Migration of CUDNN_DATA_BOOLEAN is not supported.
    */
    /*
    DPCT1007:1273: Migration of CUDNN_POINTWISE_CMP_GE is not supported.
    */
    auto rowGreaterColDesc = pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_CMP_GE);

    // Create a greater than equal to Node.
    auto rowGreaterCol_op = binary_pw_op_create(rowIndexTensor, columnIndexTensor, causalMaskTensor, rowGreaterColDesc);

    /////////////////// Apply the mask //////////////////////////

    // Define the binary select to perform masking descriptor
    /*
    DPCT1007:1274: Migration of CUDNN_POINTWISE_BINARY_SELECT is not supported.
    */
    auto maskDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_BINARY_SELECT);

    // Create a binary select Node.
    auto mask_op =
        ternary_pw_op_create(prevBlockOutputTensor, maskValTensor, causalMaskTensor, maskOutputTensor, maskDesc);

    ops.push_back(std::move(genIndexRow_op));
    ops.push_back(std::move(genIndexColumn_op));
    ops.push_back(std::move(rowGreaterCol_op));
    ops.push_back(std::move(mask_op));

    return maskOutputTensor;
}

static cudnn_frontend::Tensor
createSoftmaxForward(int64_t b,
                     int64_t h,
                     int64_t s_q,
                     int64_t s_kv,
                     bool isTraining,
                     std::vector<cudnn_frontend::Operation>& ops,
                     cudnn_frontend::Tensor& sAfterMaskTensor) {
    int64_t afterBMM1_dim[4]    = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t afterReduction_dim[4]    = {b, h, s_q, 1};
    int64_t afterReduction_stride[4] = {h * s_q, s_q, 1, 1};

    // max (x)
    auto afterMaxReductionTensor = tensor_create(dpct::library_data_t::real_float,
                                                 VIRTUAL_ID + 150,
                                                 afterReduction_dim,
                                                 afterReduction_stride,
                                                 true,
                                                 false);  // is virtual

    // x - max(x)
    auto afterSubtractionTensor = tensor_create(dpct::library_data_t::real_float,
                                                VIRTUAL_ID + 151,
                                                afterBMM1_dim,
                                                afterBMM1_stride,
                                                true,
                                                false);  // is virtual

    // e^(x - max(x))
    auto afterExponentTensor = tensor_create(dpct::library_data_t::real_float,
                                             VIRTUAL_ID + 152,
                                             afterBMM1_dim,
                                             afterBMM1_stride,
                                             true,
                                             false);  // is virtual;

    // sum (e^(x - max(x)))
    auto afterAddReductionTensor = tensor_create(dpct::library_data_t::real_float,
                                                 VIRTUAL_ID + 153,
                                                 afterReduction_dim,
                                                 afterReduction_stride,
                                                 true,
                                                 false);  // is virtual

    // log (sum (e^(x - max(x))))
    auto afterLogLTensor = tensor_create(
        dpct::library_data_t::real_float, VIRTUAL_ID + 154, afterReduction_dim, afterReduction_stride, true, false);

    // M + log (sum (e^(x - max(x))))
    auto softmaxStatsTensor = tensor_create(dpct::library_data_t::real_float,
                                            S_STATS_ID,
                                            afterReduction_dim,
                                            afterReduction_stride,
                                            !isTraining,
                                            false);  // not virtual if training is true, virtual if training is false

    // divide (e/ sum(e))
    auto afterSoftmaxTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, afterBMM1_dim)
                                  .setStride(4, afterBMM1_stride)
                                  .setId(VIRTUAL_ID + 156)
                                  .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                  .setDataType(dpct::library_data_t::real_float)
                                  .setVirtual(true)
                                  .setByValue(false)
                                  .setReorderType(cudnn_frontend::TensorReordering_t::F16x16)
                                  .build();

    // Define the reduction descriptor
    auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(dpct::library_data_t::real_float)
                                .setReductionOp(dpct::dnnl::reduction_op::max)
                                .build();
    std::cout << reductionMaxDesc.describe() << std::endl;

    // Create a reduction max Node.
    /*
    DPCT1007:1275: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
    */
    auto reductionMax_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                               .setxDesc(sAfterMaskTensor)
                               .setyDesc(afterMaxReductionTensor)
                               .setreductionDesc(reductionMaxDesc)
                               .build();
    std::cout << reductionMax_op.describe() << std::endl;

    // Define the subtract descriptor
    /*
    DPCT1007:1276: Migration of CUDNN_POINTWISE_SUB is not supported.
    */
    auto subtractDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_SUB);

    // Create a subtract Node.
    auto subtract_op =
        binary_pw_op_create(sAfterMaskTensor, afterMaxReductionTensor, afterSubtractionTensor, subtractDesc);

    // Define the exponent descriptor
    /*
    DPCT1007:1277: Migration of CUDNN_POINTWISE_EXP is not supported.
    */
    auto exponentDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_EXP);

    // Create a exponent Node.
    auto exponent_op = unary_pw_op_create(afterSubtractionTensor, afterExponentTensor, exponentDesc);

    // Define the reduction descriptor
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(dpct::library_data_t::real_float)
                                .setReductionOp(dpct::dnnl::reduction_op::sum)
                                .build();
    std::cout << reductionAddDesc.describe() << std::endl;

    // Create a reduction add Node.
    /*
    DPCT1007:1278: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
    */
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                               .setxDesc(afterExponentTensor)
                               .setyDesc(afterAddReductionTensor)
                               .setreductionDesc(reductionAddDesc)
                               .build();

    std::cout << reductionAdd_op.describe() << std::endl;

    // Create log descriptor
    /*
    DPCT1007:1279: Migration of CUDNN_POINTWISE_LOG is not supported.
    */
    auto logDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_LOG);

    // Create log node
    auto log_op = unary_pw_op_create(afterAddReductionTensor, afterLogLTensor, logDesc);

    // Create add descriptor
    /*
    DPCT1007:1280: Migration of CUDNN_POINTWISE_ADD is not supported.
    */
    auto addDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_ADD);

    // Create add node
    auto add_op = binary_pw_op_create(afterMaxReductionTensor, afterLogLTensor, softmaxStatsTensor, addDesc);

    // Define the division descriptor
    /*
    DPCT1007:1281: Migration of CUDNN_POINTWISE_DIV is not supported.
    */
    auto divisionDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_DIV);

    // Create a subtract Node.
    auto division_op =
        binary_pw_op_create(afterExponentTensor, afterAddReductionTensor, afterSoftmaxTensor, divisionDesc);

    ops.push_back(std::move(reductionMax_op));
    ops.push_back(std::move(subtract_op));
    ops.push_back(std::move(exponent_op));
    ops.push_back(std::move(reductionAdd_op));
    ops.push_back(std::move(log_op));
    ops.push_back(std::move(add_op));
    ops.push_back(std::move(division_op));

    return afterSoftmaxTensor;
}

static cudnn_frontend::Tensor
createDropoutForward(int64_t b,
                     int64_t h,
                     int64_t s_q,
                     int64_t s_kv,
                     int64_t d,
                     double probability,
                     dpct::library_data_t tensorType,
                     std::vector<cudnn_frontend::Operation>& ops,
                     cudnn_frontend::Tensor& afterSoftmaxTensor) {
    CUDNN_FRONTEND_UNUSED(d);

    cudnn_frontend::throw_if(ops.size() == 0, "Dropout DAG constructed incorrectly as the first one", 2000);

    int64_t afterBMM1_dim[4]    = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim[4]    = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    auto dropoutSeed =
        /*
        DPCT1007:1282: Migration of CUDNN_DATA_INT64 is not supported.
        */
        tensor_create(CUDNN_DATA_INT64, D_SEED_ID, scale_dim, scale_stride, false, false);  // not virtual
    auto dropoutOffset =
        /*
        DPCT1007:1283: Migration of CUDNN_DATA_INT64 is not supported.
        */
        tensor_create(CUDNN_DATA_INT64, D_OFFSET_ID, scale_dim, scale_stride, false, false);  // not virtual

    // mask for the dropout
    auto dropoutMaskTensor = tensor_create(dpct::library_data_t::real_float,
                                           VIRTUAL_ID + 200,
                                           afterBMM1_dim,
                                           afterBMM1_stride,
                                           true,
                                           false);  // is virtual
    // after dropout tensor
    auto afterDropoutTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, afterBMM1_dim)
                                  .setStride(4, afterBMM1_stride)
                                  .setId(VIRTUAL_ID + 201)
                                  .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                  .setDataType(tensorType)
                                  .setVirtual(true)
                                  .setByValue(false)
                                  .setReorderType(cudnn_frontend::TensorReordering_t::F16x16)
                                  .build();
    // scale after dropout
    auto scaleDropoutTensor =
        tensor_create(tensorType, D_CONST_ID, scale_dim, scale_stride, false, true);  // is by value
    // after Scale
    auto afterScaleTensor =
        tensor_create(tensorType, VIRTUAL_ID + 202, afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                       /*
                       DPCT1007:1284: Migration of CUDNN_RNG_DISTRIBUTION_BERNOULLI is not supported.
                       */
                       .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                       .setBernoulliDistProbability(1.0 - probability)
                       .build();
    std::cout << rngDesc.describe() << std::endl;

    // Create a rng Node.
    /*
    DPCT1007:1285: Migration of CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR is not supported.
    */
    auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                      .setyDesc(dropoutMaskTensor)
                      .setSeedDesc(dropoutSeed)
                      .setOffsetDesc(dropoutOffset)
                      .setRngDesc(rngDesc)
                      .build();

    std::cout << rng_op.describe() << std::endl;

    // Define the multiply mask descriptor
    /*
    DPCT1007:1286: Migration of CUDNN_POINTWISE_MUL is not supported.
    */
    auto maskMulDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto maskMul_op = binary_pw_op_create(afterSoftmaxTensor, dropoutMaskTensor, afterDropoutTensor, maskMulDesc);

    // Define the multiply scale descriptor
    /*
    DPCT1007:1287: Migration of CUDNN_POINTWISE_MUL is not supported.
    */
    auto scaleMulDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);

    // Create a multiply scale Node.
    auto scaleMul_op = binary_pw_op_create(afterDropoutTensor, scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

    ops.push_back(std::move(rng_op));
    ops.push_back(std::move(maskMul_op));
    ops.push_back(std::move(scaleMul_op));

    return afterScaleTensor;
}

static cudnn_frontend::Tensor
createDropoutBackward(int64_t b,
                      int64_t h,
                      int64_t s_q,
                      int64_t s_kv,
                      int64_t d,
                      double probability,
                      dpct::library_data_t tensorType,
                      std::vector<cudnn_frontend::Operation>& ops,
                      cudnn_frontend::Tensor& afterSoftmaxTensor,
                      cudnn_frontend::Tensor& dropoutMaskTensor) {
    CUDNN_FRONTEND_UNUSED(d);

    cudnn_frontend::throw_if(ops.size() == 0, "Dropout DAG constructed incorrectly as the first one", 2000);

    int64_t afterBMM1_dim[4]    = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim[4]    = {1, 1, 1, 1};
    int64_t scale_stride[4] = {1, 1, 1, 1};

    auto dropoutSeed =
        /*
        DPCT1007:1288: Migration of CUDNN_DATA_INT64 is not supported.
        */
        tensor_create(CUDNN_DATA_INT64, D_SEED_ID, scale_dim, scale_stride, false, false);  // not virtual
    auto dropoutOffset =
        /*
        DPCT1007:1289: Migration of CUDNN_DATA_INT64 is not supported.
        */
        tensor_create(CUDNN_DATA_INT64, D_OFFSET_ID, scale_dim, scale_stride, false, false);  // not virtual

    // after dropout tensor
    auto afterDropoutTensor = cudnn_frontend::TensorBuilder()
                                  .setDim(4, afterBMM1_dim)
                                  .setStride(4, afterBMM1_stride)
                                  .setId(VIRTUAL_ID + 201)
                                  .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                                  .setDataType(tensorType)
                                  .setVirtual(true)
                                  .setByValue(false)
                                  .setReorderType(cudnn_frontend::TensorReordering_t::F16x16)
                                  .build();
    // scale after dropout
    auto scaleDropoutTensor =
        tensor_create(tensorType, D_CONST_ID, scale_dim, scale_stride, false, true);  // is by value
    // after Scale
    auto afterScaleTensor =
        tensor_create(tensorType, VIRTUAL_ID + 202, afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                       /*
                       DPCT1007:1290: Migration of CUDNN_RNG_DISTRIBUTION_BERNOULLI is not supported.
                       */
                       .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                       .setBernoulliDistProbability(1.0 - probability)
                       .build();
    std::cout << rngDesc.describe() << std::endl;

    // Create a rng Node.
    /*
    DPCT1007:1291: Migration of CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR is not supported.
    */
    auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                      .setyDesc(dropoutMaskTensor)
                      .setSeedDesc(dropoutSeed)
                      .setOffsetDesc(dropoutOffset)
                      .setRngDesc(rngDesc)
                      .build();

    std::cout << rng_op.describe() << std::endl;

    // Define the multiply mask descriptor
    /*
    DPCT1007:1292: Migration of CUDNN_POINTWISE_MUL is not supported.
    */
    auto maskMulDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto maskMul_op = binary_pw_op_create(afterSoftmaxTensor, dropoutMaskTensor, afterDropoutTensor, maskMulDesc);

    // Define the multiply scale descriptor
    /*
    DPCT1007:1293: Migration of CUDNN_POINTWISE_MUL is not supported.
    */
    auto scaleMulDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);

    // Create a multiply scale Node.
    auto scaleMul_op = binary_pw_op_create(afterDropoutTensor, scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

    ops.push_back(std::move(rng_op));
    ops.push_back(std::move(maskMul_op));
    ops.push_back(std::move(scaleMul_op));

    return afterScaleTensor;
}

static void
createSVBMM(int64_t b,
            int64_t h,
            int64_t s_q,
            int64_t s_kv,
            int64_t d,
            MHA_Layout layout,
            dpct::library_data_t tensorType,
            std::vector<cudnn_frontend::Operation>& ops,
            cudnn_frontend::Tensor const& afterScaleDropoutTensor) {
    cudnn_frontend::throw_if(ops.size() == 0, "BMM2 op constructed incorrectly as the first one", 2000);

    int64_t v_dim[4] = {b, h, s_kv, d};
    int64_t v_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);

    int64_t o_dim[4] = {b, h, s_q, d};
    int64_t o_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

    auto vTensor = tensor_create(tensorType, V_ID, v_dim, v_stride, false, false);
    // second GEMM output
    auto oTensor = tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);

    // Define the matmul 2 desc
    auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
    std::cout << matmul_2_Desc.describe() << std::endl;

    // Create a matmul 2 Node
    /*
    DPCT1007:1294: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
    */
    auto matmul_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                          .setaMatDesc(afterScaleDropoutTensor)
                          .setbMatDesc(vTensor)
                          .setcMatDesc(oTensor)
                          .setmatmulDesc(matmul_2_Desc)
                          .build();

    std::cout << matmul_op2.describe() << std::endl;

    ops.push_back(std::move(matmul_op2));
}

static cudnn_frontend::Tensor
createSoftmaxBackward(int64_t b,
                      int64_t h,
                      int64_t s_q,
                      int64_t s_kv,
                      int64_t d,
                      MHA_Layout layout,
                      dpct::library_data_t tensorType,
                      std::vector<cudnn_frontend::Operation>& ops,
                      cudnn_frontend::Tensor& yTensor,
                      cudnn_frontend::Tensor& dyTensor) {
    CUDNN_FRONTEND_UNUSED(tensorType);

    cudnn_frontend::throw_if(ops.size() == 0, "Softmax backward constructed incorrectly as the first one", 2000);

    int64_t p_dim[4] = {b, h, s_q, s_kv};
    int64_t p_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, p_stride, layout, MHA_Matrix::S_Matrix);

    int64_t p_reduction_dim[4] = {b, h, s_q, 1};
    int64_t p_reduction_stride[4];

    p_reduction_stride[3] = 1;
    p_reduction_stride[2] = 1;
    p_reduction_stride[1] = s_q;
    p_reduction_stride[0] = s_q * h;

    int64_t const_dim[4]    = {1, 1, 1, 1};
    int64_t const_stride[4] = {1, 1, 1, 1};

    // creating all tensors
    auto softmaxScaleTensor =
        tensor_create(dpct::library_data_t::real_float, S_CONST_ID, const_dim, const_stride, false, true);
    auto dyMulYTensor = tensor_create(dpct::library_data_t::real_float, VIRTUAL_ID + 250, p_dim, p_stride, true, false);
    auto dxAfterReductionTensor = tensor_create(
        dpct::library_data_t::real_float, VIRTUAL_ID + 251, p_reduction_dim, p_reduction_stride, true, false);
    auto dxAfterSubtractionTensor =
        tensor_create(dpct::library_data_t::real_float, VIRTUAL_ID + 252, p_dim, p_stride, true, false);
    auto dxUnscaleTensor =
        tensor_create(dpct::library_data_t::real_float, VIRTUAL_ID + 253, p_dim, p_stride, true, false);
    auto dxTensor = tensor_create(dpct::library_data_t::real_float, VIRTUAL_ID + 254, p_dim, p_stride, true, false);

    // creating all ops
    // mul (y * dy)
    /*
    DPCT1007:1295: Migration of CUDNN_POINTWISE_MUL is not supported.
    */
    auto mul_1_desc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);
    auto mul_1_op   = binary_pw_op_create(yTensor, dyTensor, dyMulYTensor, mul_1_desc);

    // reduction add sum (y * dy)
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(dpct::library_data_t::real_float)
                                .setReductionOp(dpct::dnnl::reduction_op::sum)
                                .build();
    std::cout << reductionAddDesc.describe() << std::endl;

    /*
    DPCT1007:1296: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
    */
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                               .setxDesc(dyMulYTensor)
                               .setyDesc(dxAfterReductionTensor)
                               .setreductionDesc(reductionAddDesc)
                               .build();

    std::cout << reductionAdd_op.describe() << std::endl;

    // subtraction (dy - sum(y * dy))
    /*
    DPCT1007:1297: Migration of CUDNN_POINTWISE_SUB is not supported.
    */
    auto sub_0_desc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_SUB);
    auto sub_0_op   = binary_pw_op_create(dyTensor, dxAfterReductionTensor, dxAfterSubtractionTensor, sub_0_desc);

    // mul (y * (dy - sum(y * dy)))
    /*
    DPCT1007:1298: Migration of CUDNN_POINTWISE_MUL is not supported.
    */
    auto mul_2_desc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);
    auto mul_2_op   = binary_pw_op_create(yTensor, dxAfterSubtractionTensor, dxUnscaleTensor, mul_2_desc);

    // mul (scale * dx)
    /*
    DPCT1007:1299: Migration of CUDNN_POINTWISE_MUL is not supported.
    */
    auto mul_3_desc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);
    auto mul_3_op   = binary_pw_op_create(dxUnscaleTensor, softmaxScaleTensor, dxTensor, mul_3_desc);

    ops.push_back(std::move(mul_1_op));
    ops.push_back(std::move(reductionAdd_op));
    ops.push_back(std::move(sub_0_op));
    ops.push_back(std::move(mul_2_op));
    ops.push_back(std::move(mul_3_op));

    return dxTensor;
}

void
run_f16_flash_attention_fprop(int64_t b,
                              int64_t h,
                              int64_t s_q,
                              int64_t s_kv,
                              int64_t d,
                              MHA_Layout layout,
                              float scaling_factor,
                              bool isTraining,
                              double dropout_probability,
                              void* devPtrQ,
                              void* devPtrK,
                              void* devPtrV,
                              void* devPtrSoftmaxStats,
                              void* devPtrO,
                              void* devPtrDropoutSeed,
                              void* devPtrDropoutOffset,
                              dpct::library_data_t tensorType) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        std::vector<cudnn_frontend::Operation const*> all_ops;
        std::vector<cudnn_frontend::Operation> ops;
        std::set<std::pair<uint64_t, void*>> data_ptrs;

        // Q * K^T
        auto sTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, ops);

        // Q * K^T * bmmScale
        auto sScaleTensor = createScale(b, h, s_q, s_kv, d, layout, dpct::library_data_t::real_float, sTensor, ops);

        // Causual mask
        float negInfinity     = -1.0E+20f;  // change this if you have access to float_min
        auto sAfterMaskTensor = createCausalMask(b, h, s_q, s_kv, d, layout, tensorType, ops, sScaleTensor);

        cudnn_frontend::throw_if(
            dropout_probability != 0.0f && !isTraining, "Dropout probability should be 0.0f for inference mode", 2000);
        cudnn_frontend::throw_if(dropout_probability == 1.0f, "Dropout probability cannot be 1.0", 2000);

        // needs to be bf16 (Please change)
        half1 scale_dropout = cpu_float2half_rn(static_cast<float>(1 / (1 - dropout_probability)));

        auto softmax_output = createSoftmaxForward(b, h, s_q, s_kv, isTraining, ops, sAfterMaskTensor);

        // Dropout(softmax)
        auto dropout_output =
            createDropoutForward(b, h, s_q, s_kv, d, dropout_probability, tensorType, ops, softmax_output);
        createSVBMM(b, h, s_q, s_kv, d, layout, tensorType, ops, dropout_output);

        std::cout << "Total ops created: " << ops.size() << std::endl;

        for (unsigned int i = 0; i < ops.size(); i++) {
            all_ops.push_back(&ops[i]);
        }

        // Create an Operation Graph
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(all_ops.size(), all_ops.data())
                           .build();

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<1>(
            {"heuristics_instant"}, opGraph, ::allowAllConfig, filtered_configs, true);

        if (filtered_configs.size() == 0) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3000, "run_mha_fprop: No config returned by the heuristics");
        }

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device(workspace_size, q_ct1)));
        }

        // add all the data pointers to be used in the variant pack
        data_ptrs.insert(std::pair<uint64_t, void*>(Q_ID, devPtrQ));
        data_ptrs.insert(std::pair<uint64_t, void*>(K_ID, devPtrK));
        data_ptrs.insert(std::pair<uint64_t, void*>(V_ID, devPtrV));
        data_ptrs.insert(std::pair<uint64_t, void*>(MASK_VAL_ID, &negInfinity));
        data_ptrs.insert(std::pair<uint64_t, void*>(S_CONST_ID, &scaling_factor));
        data_ptrs.insert(std::pair<uint64_t, void*>(O_ID, devPtrO));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_SEED_ID, devPtrDropoutSeed));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_OFFSET_ID, devPtrDropoutOffset));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_CONST_ID, &scale_dropout));

        // If training mode, we write out softmax stats
        if (isTraining) {
            data_ptrs.insert(std::pair<uint64_t, void*>(S_STATS_ID, devPtrSoftmaxStats));
        }

        auto variantPack =
            cudnn_frontend::VariantPackBuilder().setWorkspacePointer(workspace_ptr).setDataPointers(data_ptrs).build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1300: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudaErr(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1301: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
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

        // this example is only for GA100 cards and GH100 cards
        /*
        DPCT1005:1302: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (!((prop.get_major_version() == 8 && prop.get_minor_version() == 0) ||
              (prop.get_major_version() == 9 && prop.get_minor_version() == 0 && CUDNN_VERSION >= 8900)) &&
            (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for GA100 (cuDNN >= 8900) and GH100 (cuDNN >= 8900) GPUs"
                      << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}

void
run_f16_flash_attention_bprop(int64_t b,
                              int64_t h,
                              int64_t s_q,
                              int64_t s_kv,
                              int64_t d,
                              MHA_Layout layout,
                              float scaling_factor,
                              float dropout_probability,
                              void* devPtrQ,
                              void* devPtrKTranspose,
                              void* devPtrVTranspose,
                              void* devPtrO,
                              void* devPtrSoftmaxStats,
                              void* devPtrSoftmaxSum,
                              void* devPtrdQAccumulator,
                              void* devPtrdQ,
                              void* devPtrdK,
                              void* devPtrdV,
                              void* devPtrdO,
                              void* devPtrDropoutSeed,
                              void* devPtrDropoutOffset,
                              dpct::library_data_t tensorType) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
    dpct::dnnl::engine_ext handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(DPCT_CHECK_ERROR(handle_.create_engine()));

        std::vector<cudnn_frontend::Operation const*> all_ops;
        std::vector<cudnn_frontend::Operation> ops;
        std::set<std::pair<uint64_t, void*>> data_ptrs;

        // Creates the necessary tensor descriptors
        int64_t q_dim[4] = {b, h, s_q, d};
        int64_t q_stride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

        int64_t k_transpose_dim[4] = {b, h, d, s_kv};
        int64_t k_transpose_stride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, k_transpose_stride, layout, MHA_Matrix::K_Matrix_Transpose);

        int64_t v_transpose_dim[4] = {b, h, d, s_kv};
        int64_t v_transpose_stride[4];
        generateMHAStrides(b,
                           h,
                           s_q,
                           s_kv,
                           d,
                           v_transpose_stride,
                           layout,
                           MHA_Matrix::V_Matrix_Transpose);  // type is correct as V is transposed

        int64_t p_dim[4] = {b, h, s_q, s_kv};
        int64_t p_stride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, p_stride, layout, MHA_Matrix::S_Matrix);

        int64_t p_transpose_dim[4] = {b, h, s_kv, s_q};
        int64_t p_transpose_stride[4];
        p_transpose_stride[0] = p_stride[0];
        p_transpose_stride[1] = p_stride[1];
        p_transpose_stride[2] = p_stride[3];
        p_transpose_stride[3] = p_stride[2];

        int64_t o_dim[4] = {b, h, s_q, d};
        int64_t o_stride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

        int64_t dqAccum_dim[4] = {b, h, s_q, d};
        int64_t dqAccum_stride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, dqAccum_stride, layout, MHA_Matrix::O_Matrix);

        int64_t scale_dim[4]    = {1, 1, 1, 1};
        int64_t scale_stride[4] = {1, 1, 1, 1};

        /*******************************************************************************
         *                          Dot product dO * O
         *///////////////////////////////////////////////////////////////////////////////
        // output and gradient of the output
        auto oTensor  = tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);
        auto dOTensor = tensor_create(tensorType, dO_ID, o_dim, o_stride, false, false);

        auto dotProductTensor =
            tensor_create(dpct::library_data_t::real_float, VIRTUAL_ID, o_dim, o_stride, true, false);  // is virtual
        // Create pointwise mul
        /*
        DPCT1007:1303: Migration of CUDNN_POINTWISE_MUL is not supported.
        */
        auto multiplyDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_MUL);
        // do * O
        auto dotProductOp = binary_pw_op_create(dOTensor, oTensor, dotProductTensor, multiplyDesc);
        ops.push_back(std::move(dotProductOp));

        /*******************************************************************************
         *                         Reduction(dO * O)
         *///////////////////////////////////////////////////////////////////////////////

        int64_t reduction_dim[4]    = {b, h, s_q, 1};
        int64_t reduction_stride[4] = {h * s_q, s_q, 1, 1};
        // reduction(dO * O)
        auto afterReductionTensor = tensor_create(dpct::library_data_t::real_float,
                                                  VIRTUAL_ID + 1,
                                                  reduction_dim,
                                                  reduction_stride,
                                                  true,
                                                  false);  // is virtual
        auto reductionAddDesc     = cudnn_frontend::ReductionDescBuilder()
                                    .setComputeType(dpct::library_data_t::real_float)
                                    .setReductionOp(dpct::dnnl::reduction_op::sum)
                                    .build();
        std::cout << reductionAddDesc.describe() << std::endl;

        // Create a reduction add Node.
        /*
        DPCT1007:1304: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
        */
        auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                   .setxDesc(dotProductTensor)
                                   .setyDesc(afterReductionTensor)
                                   .setreductionDesc(reductionAddDesc)
                                   .build();
        std::cout << reductionAdd_op.describe() << std::endl;
        ops.push_back(std::move(reductionAdd_op));

        /*******************************************************************************
         *                        reduction(dO * O) * scale prob -> softmaxSum
         *///////////////////////////////////////////////////////////////////////////////
        auto softmaxSumTensor = tensor_create(
            dpct::library_data_t::real_float, S_SUM_ID, reduction_dim, reduction_stride, false, false);  // not virtual
        auto scaleProbTensor = tensor_create(
            dpct::library_data_t::real_float, SCALE_PROB, scale_dim, scale_stride, false, true);  // not virtual
        auto softmaxSumOp = binary_pw_op_create(afterReductionTensor, scaleProbTensor, softmaxSumTensor, multiplyDesc);
        ops.push_back(std::move(softmaxSumOp));

        /*******************************************************************************
         *                        Q @ K.T -> P
         *///////////////////////////////////////////////////////////////////////////////

        // Inputs from fprop
        auto qTensor          = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
        auto kTransposeTensor = tensor_create(tensorType, K_ID, k_transpose_dim, k_transpose_stride, false, false);
        auto pTensor          = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 2, p_dim, p_stride, true, false);  // is virtual

        // matmul to calculate dvTensor
        auto matmul_0_Desc =
            cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
        std::cout << matmul_0_Desc.describe() << std::endl;

        /*
        DPCT1007:1305: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
        */
        auto matmul_op0 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                              .setaMatDesc(qTensor)
                              .setbMatDesc(kTransposeTensor)
                              .setcMatDesc(pTensor)
                              .setmatmulDesc(matmul_0_Desc)
                              .build();

        std::cout << matmul_op0.describe() << std::endl;
        ops.push_back(std::move(matmul_op0));

        /*******************************************************************************
         *                        P * bmmScale -> pAfterScale
         *///////////////////////////////////////////////////////////////////////////////
        auto bmmScaleTensor    = tensor_create(dpct::library_data_t::real_float,
                                            S_CONST_ID,
                                            scale_dim,
                                            scale_stride,
                                            false,
                                            true);  // not virtual and by value
        auto pAfterScaleTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 2000, p_dim, p_stride, true, false);  // virtual
        auto scaleOp = binary_pw_op_create(pTensor, bmmScaleTensor, pAfterScaleTensor, multiplyDesc);
        ops.push_back(std::move(scaleOp));

        /*******************************************************************************
         *                          Causal masking -> pAfterMaskTensor
         *///////////////////////////////////////////////////////////////////////////////
        float negInfinity     = -1.0E+20f;  // change this if you have access to float_min
        auto pAfterMaskTensor = createCausalMask(b, h, s_q, s_kv, d, layout, tensorType, ops, pAfterScaleTensor);

        /*******************************************************************************
         *                          pAfterMaskTensor - softmaxStats -> pAfterSubtract
         *///////////////////////////////////////////////////////////////////////////////
        auto pAfterSubtractTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 3, p_dim, p_stride, true, false);  // is virtual
        auto softmaxStatsTensor = tensor_create(dpct::library_data_t::real_float,
                                                S_STATS_ID,
                                                reduction_dim,
                                                reduction_stride,
                                                false,
                                                false);  // not virtual
        /*
        DPCT1007:1306: Migration of CUDNN_POINTWISE_SUB is not supported.
        */
        auto subtractDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_SUB);
        auto subtract_op =
            binary_pw_op_create(pAfterMaskTensor, softmaxStatsTensor, pAfterSubtractTensor, subtractDesc);
        ops.push_back(std::move(subtract_op));

        /*******************************************************************************
         *                          e^(pAfterSubtract) -> pAfterSoftmax
         *///////////////////////////////////////////////////////////////////////////////
        auto pAfterSoftmaxTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 4, p_dim, p_stride, true, false);  // is virtual
        /*
        DPCT1007:1307: Migration of CUDNN_POINTWISE_EXP is not supported.
        */
        auto expDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_EXP);
        auto exp_op  = unary_pw_op_create(pAfterSubtractTensor, pAfterSoftmaxTensor, expDesc);
        ops.push_back(std::move(exp_op));

        /*******************************************************************************
         *                          Dropout -> afterScaleDropout
         *///////////////////////////////////////////////////////////////////////////////
        // mask for the dropout. used in rng and dS
        auto dropoutMaskTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 5, p_dim, p_stride, true, false);  // is virtual
        auto afterScaleDropoutTensor = createDropoutBackward(
            b, h, s_q, s_kv, d, dropout_probability, tensorType, ops, pAfterSoftmaxTensor, dropoutMaskTensor);

        /*******************************************************************************
         *                          afterScaleDropout -> sTransposeTensor
         *///////////////////////////////////////////////////////////////////////////////
        auto sTransposeTensor =
            tensor_create(tensorType, VIRTUAL_ID + 6, p_transpose_dim, p_transpose_stride, true, false);  // is virtual
        /*
        DPCT1007:1308: Migration of CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR is not supported.
        */
        auto reshape_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                              .setxDesc(afterScaleDropoutTensor)
                              .setyDesc(sTransposeTensor)
                              .build();
        ops.push_back(std::move(reshape_op));

        // Outputs of bprop
        int64_t dq_dim[4] = {b, h, s_q, d};
        int64_t dq_stride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, dq_stride, layout, MHA_Matrix::Q_Matrix);

        int64_t dk_dim[4] = {b, h, s_kv, d};
        int64_t dk_stride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, dk_stride, layout, MHA_Matrix::K_Matrix);

        int64_t dv_dim[4] = {b, h, s_kv, d};
        int64_t dv_stride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, dv_stride, layout, MHA_Matrix::V_Matrix);
        // Outputs of backprop
        auto dQTensor = tensor_create(tensorType, dQ_ID, dq_dim, dq_stride, false, false);
        auto dKTensor = tensor_create(tensorType, dK_ID, dk_dim, dk_stride, false, false);
        auto dVTensor = tensor_create(tensorType, dV_ID, dv_dim, dv_stride, false, false);  // not virtual

        /*******************************************************************************
         *                          sTransposeTensor @ dO -> dV
         *///////////////////////////////////////////////////////////////////////////////
        auto matmul_1_Desc =
            cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
        std::cout << matmul_1_Desc.describe() << std::endl;

        /*
        DPCT1007:1309: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
        */
        auto matmul_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                              .setaMatDesc(sTransposeTensor)
                              .setbMatDesc(dOTensor)
                              .setcMatDesc(dVTensor)
                              .setmatmulDesc(matmul_1_Desc)
                              .build();

        std::cout << matmul_op1.describe() << std::endl;
        ops.push_back(std::move(matmul_op1));

        /*******************************************************************************
         *                          dO @ V.T -> dS
         *///////////////////////////////////////////////////////////////////////////////
        auto vTransposeTensor = tensor_create(tensorType, V_ID, v_transpose_dim, v_transpose_stride, false, false);
        auto dSTensor         = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 7, p_dim, p_stride, true, false);  // is virtual

        auto matmul_2_Desc =
            cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
        std::cout << matmul_1_Desc.describe() << std::endl;

        /*
        DPCT1007:1310: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
        */
        auto matmul_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                              .setaMatDesc(dOTensor)
                              .setbMatDesc(vTransposeTensor)
                              .setcMatDesc(dSTensor)
                              .setmatmulDesc(matmul_2_Desc)
                              .build();

        std::cout << matmul_op2.describe() << std::endl;
        ops.push_back(std::move(matmul_op2));

        /*******************************************************************************
         *                          dS * dropoutMask -> dSAfterDropout
         *///////////////////////////////////////////////////////////////////////////////
        auto dSAfterDropoutTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 8, p_dim, p_stride, true, false);  // is virtual
        auto multiply_op = binary_pw_op_create(dSTensor, dropoutMaskTensor, dSAfterDropoutTensor, multiplyDesc);
        ops.push_back(std::move(multiply_op));

        /*******************************************************************************
         *                          dSAfterDropout - softmaxSum -> dsAfterSubtract
         *///////////////////////////////////////////////////////////////////////////////
        auto dsAfterSubtractTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 9, p_dim, p_stride, true, false);  // is virtual
        auto subtract_op2 =
            binary_pw_op_create(dSAfterDropoutTensor, softmaxSumTensor, dsAfterSubtractTensor, subtractDesc);
        ops.push_back(std::move(subtract_op2));

        /*******************************************************************************
         *                          dsAfterSubtract * afterSoftmax -> dP
         *///////////////////////////////////////////////////////////////////////////////
        auto dPTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 10, p_dim, p_stride, true, false);  // is virtual
        auto multiply_op2 = binary_pw_op_create(dsAfterSubtractTensor, pAfterSoftmaxTensor, dPTensor, multiplyDesc);
        ops.push_back(std::move(multiply_op2));

        /*******************************************************************************
         *                          dP * scaleDropout -> dPAfterDropoutScale
         *///////////////////////////////////////////////////////////////////////////////
        auto dPAfterDropoutScaleTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 11, p_dim, p_stride, true, false);  // is virtual
        // needs to be bf16 (Please change)
        half1 scale_dropout = cpu_float2half_rn(static_cast<float>(1 / (1 - dropout_probability)));
        auto scaleDropoutTensor =
            tensor_create(tensorType, D_CONST_ID, scale_dim, scale_stride, false, true);  // is by value
        auto multiply_op3 = binary_pw_op_create(dPTensor, scaleDropoutTensor, dPAfterDropoutScaleTensor, multiplyDesc);
        ops.push_back(std::move(multiply_op3));

        /*******************************************************************************
         *                          dPAfterDropoutScale * bmmScale -> dPScaledTensor
         *///////////////////////////////////////////////////////////////////////////////
        auto dPScaledTensor = tensor_create(
            dpct::library_data_t::real_float, VIRTUAL_ID + 12, p_dim, p_stride, true, false);  // is virtual
        auto multiply_op4 =
            binary_pw_op_create(dPAfterDropoutScaleTensor, bmmScaleTensor, dPScaledTensor, multiplyDesc);
        ops.push_back(std::move(multiply_op4));

        /*******************************************************************************
         *                          K.T -> K
         *///////////////////////////////////////////////////////////////////////////////
        int64_t kDim[4] = {b, h, s_kv, d};
        int64_t kStride[4];
        generateMHAStrides(b, h, s_q, s_kv, d, kStride, layout, MHA_Matrix::K_Matrix);
        auto kTensor     = tensor_create(tensorType, VIRTUAL_ID + 13, kDim, kStride, true, false);  // is virtual
        /*
        DPCT1007:1311: Migration of CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR is not supported.
        */
        auto reshape_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                               .setxDesc(kTransposeTensor)
                               .setyDesc(kTensor)
                               .build();
        std::cout << reshape_op2.describe() << std::endl;
        ops.push_back(std::move(reshape_op2));

        /*******************************************************************************
         *                          dP @ K -> dqAccumTensor
         *///////////////////////////////////////////////////////////////////////////////
        auto dqAccumTensor =
            cudnn_frontend::TensorBuilder()
                .setDim(4, dqAccum_dim)
                .setStride(4, dqAccum_stride)
                .setId(dQ_ACCUM_ID)
                .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dpct::library_data_t::real_float)
                .setVirtual(false)
                .setByValue(false)
                .setReorderType(cudnn_frontend::TensorReordering_t::F16x16)  // dqAccum has reorder type
                .build();

        auto matmul_3_Desc =
            cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
        /*
        DPCT1007:1312: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
        */
        auto matmul_op3 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                              .setaMatDesc(dPScaledTensor)
                              .setbMatDesc(kTensor)
                              .setcMatDesc(dqAccumTensor)
                              .setmatmulDesc(matmul_3_Desc)
                              .build();

        std::cout << matmul_op3.describe() << std::endl;
        ops.push_back(std::move(matmul_op3));

        /*******************************************************************************
         *                          dP.T @ Q -> dK
         *///////////////////////////////////////////////////////////////////////////////

        auto dPTransposeTensor = tensor_create(dpct::library_data_t::real_float,
                                               VIRTUAL_ID + 14,
                                               p_transpose_dim,
                                               p_transpose_stride,
                                               true,
                                               false);  // is virtual
        /*
        DPCT1007:1313: Migration of CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR is not supported.
        */
        auto reshape_op3 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                               .setxDesc(dPScaledTensor)
                               .setyDesc(dPTransposeTensor)
                               .build();
        std::cout << reshape_op3.describe() << std::endl;
        ops.push_back(std::move(reshape_op3));

        auto matmul_4_Desc =
            cudnn_frontend::MatMulDescBuilder().setComputeType(dpct::library_data_t::real_float).build();
        /*
        DPCT1007:1314: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
        */
        auto matmul_op4 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                              .setaMatDesc(dPTransposeTensor)
                              .setbMatDesc(qTensor)
                              .setcMatDesc(dKTensor)
                              .setmatmulDesc(matmul_4_Desc)
                              .build();

        std::cout << matmul_op4.describe() << std::endl;
        ops.push_back(std::move(matmul_op4));

        /*******************************************************************************
         *                          dqAccumTensor @ identity -> dqTensor
         *///////////////////////////////////////////////////////////////////////////////
        /*
        DPCT1007:1315: Migration of CUDNN_POINTWISE_IDENTITY is not supported.
        */
        auto identityDesc = pw_desc_create(dpct::library_data_t::real_float, CUDNN_POINTWISE_IDENTITY);
        auto identity_op  = unary_pw_op_create(dqAccumTensor, dQTensor, identityDesc);
        ops.push_back(std::move(identity_op));

        std::cout << "Total ops created: " << ops.size() << std::endl;

        for (unsigned int i = 0; i < ops.size(); i++) {
            all_ops.push_back(&ops[i]);
        }

        // Create an Operation Graph
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(all_ops.size(), all_ops.data())
                           .build();

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<1>(
            {"heuristics_instant"}, opGraph, ::allowAllConfig, filtered_configs, true);

        if (filtered_configs.size() == 0) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, 3000, "run_mha_bprop: No config returned by the heuristics");
        }

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle_)
                        .setEngineConfig(filtered_configs[0], opGraph.getTag())
                        .build();

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(workspace_ptr = (void*)sycl::malloc_device(workspace_size, q_ct1)));
        }

        // add all the data pointers to be used in the variant pack
        data_ptrs.insert(std::pair<uint64_t, void*>(dQ_ID, devPtrdQ));
        data_ptrs.insert(std::pair<uint64_t, void*>(dQ_ACCUM_ID, devPtrdQAccumulator));
        data_ptrs.insert(std::pair<uint64_t, void*>(dK_ID, devPtrdK));
        data_ptrs.insert(std::pair<uint64_t, void*>(dV_ID, devPtrdV));

        data_ptrs.insert(std::pair<uint64_t, void*>(Q_ID, devPtrQ));
        data_ptrs.insert(std::pair<uint64_t, void*>(K_ID, devPtrKTranspose));
        data_ptrs.insert(std::pair<uint64_t, void*>(V_ID, devPtrVTranspose));
        data_ptrs.insert(std::pair<uint64_t, void*>(O_ID, devPtrO));
        data_ptrs.insert(std::pair<uint64_t, void*>(dO_ID, devPtrdO));
        data_ptrs.insert(std::pair<uint64_t, void*>(S_STATS_ID, devPtrSoftmaxStats));
        data_ptrs.insert(std::pair<uint64_t, void*>(S_SUM_ID, devPtrSoftmaxSum));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_SEED_ID, devPtrDropoutSeed));
        data_ptrs.insert(std::pair<uint64_t, void*>(D_OFFSET_ID, devPtrDropoutOffset));
        data_ptrs.insert(std::pair<uint64_t, void*>(MASK_VAL_ID, &negInfinity));

        float scaleProb = 1.0f - dropout_probability;
        data_ptrs.insert(std::pair<uint64_t, void*>(D_CONST_ID, &scale_dropout));
        data_ptrs.insert(std::pair<uint64_t, void*>(S_CONST_ID, &scaling_factor));
        data_ptrs.insert(std::pair<uint64_t, void*>(SCALE_PROB, &scaleProb));

        auto variantPack =
            cudnn_frontend::VariantPackBuilder().setWorkspacePointer(workspace_ptr).setDataPointers(data_ptrs).build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        /*
        DPCT1007:1316: Migration of cudnnBackendExecute is not supported.
        */
        dpct::err1 status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudaErr(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        if (workspace_size > 0) {
            checkCudaErr(DPCT_CHECK_ERROR(dpct::dpct_free(workspace_ptr, q_ct1)));
        }

        /*
        DPCT1027:1317: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
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

        // this example is only for GA100 cards and GH100 cards
        /*
        DPCT1005:1318: The SYCL device version is different from CUDA Compute Compatibility. You may need to rewrite
        this code.
        */
        if (!((prop.get_major_version() == 8 && prop.get_minor_version() == 0) ||
              (prop.get_major_version() == 9 && prop.get_minor_version() == 0 && CUDNN_VERSION >= 8900)) &&
            (e.getCudnnStatus() == 3007 || e.getCudnnStatus() == 3000)) {
            std::cout << "Example is only supported for GA100 (cuDNN >= 8900) and GH100 (cuDNN >= 8900) GPUs"
                      << std::endl;
        } else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif