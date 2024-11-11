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

#pragma once
#include <dpct/dnnl_utils.hpp>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <catch2/catch_test_macros.hpp>

#include "fp16_dev.h"
#include "fp16_emu.h"
#include "error_util.h"

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

#define THRESHOLD 2.0e-2

enum class MHA_Layout { NOT_INTERLEAVED = 0, QKV_INTERLEAVED = 1, KV_INTERLEAVED = 2, SBH_INTERLEAVED = 3 };

enum class MHA_Matrix {
    Q_Matrix           = 0,  // queries
    K_Matrix           = 1,  // keys
    K_Matrix_Transpose = 2,  // keys tranposed
    V_Matrix           = 3,  // values
    V_Matrix_Transpose = 4,  // values transposed
    S_Matrix           = 5,  // output of GEMM1
    O_Matrix           = 6,  // final output
};

enum class MHA_Bias_Type { NO_BIAS = 0, PRE_SCALE_BIAS = 1, POST_SCALE_BIAS = 2 };

bool
is_ampere_arch();
bool
is_ada_arch();
bool
is_hopper_arch();
bool
check_device_arch_newer_than(std::string const& arch);
bool
is_arch_supported_by_cudnn();

int64_t
getFwdConvDilatedFilterDim(int64_t filterDim, int64_t dilation);
int64_t
getFwdConvPaddedImageDim(int64_t tensorDim, int64_t pad);
int64_t
getFwdConvOutputDim(int64_t tensorDim, int64_t pad, int64_t filterDim, int64_t stride, int64_t dilation);

void
generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, dpct::dnnl::memory_format_tag filterFormat);
void
generate4dTransposeStrides(const int64_t* dimA,
                           int64_t* strideA,
                           int64_t nbDims,
                           dpct::dnnl::memory_format_tag filterFormat);
void
generateMHAStrides(int64_t b,
                   int64_t h,
                   int64_t s_q,
                   int64_t s_kv,
                   int64_t d,
                   int64_t* strideA,
                   MHA_Layout layout,
                   MHA_Matrix matrix);

int64_t
checkCudaError(dpct::err0 code, const char* expr, const char* file, int line);
int64_t
checkCudnnError(dpct::err1 code, const char* expr, const char* file, int line);

void
lin2dim(int64_t id, int64_t* ids, const int64_t* dims, int64_t length);
int64_t
dim2lin(const int64_t* ids, const int64_t* strides, int64_t length);

void
initImage(float* image, int64_t imageSize);
void
initImage(half1* image, int64_t imageSize);
void
testinitImage(half1* image, int64_t imageSize, int test);
void
initImage(int8_t* image, int64_t imageSize);
void
initImage(uint8_t* image, int64_t imageSize);
void
initImage(int32_t* image, int64_t imageSize);
void
initImage(int64_t* image, int64_t imageSize);
void
initImage(bool* image, int64_t imageSize);
void
initImagePadded(int8_t* image,
                int64_t dimA[],
                int64_t dimPadded[],
                int64_t stridePadded[],
                dpct::library_data_t dataType);

void
doEpilog(float* out, int64_t idx, float alphaAcc, float beta);
void
doEpilog(half1* out, int64_t idx, float alphaAcc, float beta);
void
doEpilog(int8_t* out, int64_t idx, int32_t alphaAcc, float beta);

float
getError(float dev, float ref);
float
getError(half1 dev, half1 ref);
int8_t
getError(int8_t dev, int8_t ref);

static float
doFma(float fval, float ival, float tmp) {
    return fval * ival + tmp;
}

static float
doFma(half1 fval, half1 ival, float tmp) {
    return cpu_half2float(fval) * cpu_half2float(ival) + tmp;
}

static int32_t
doFma(int8_t fval, int8_t ival, int32_t tmp) {
    return int32_t(fval) * int32_t(ival) + tmp;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t
doFma(float fval, float ival, int32_t tmp) {
    (void)fval;
    (void)ival;
    (void)tmp;
    return 0;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t
doFma(half1 fval, half1 ival, int32_t tmp) {
    (void)fval;
    (void)ival;
    (void)tmp;
    return 0;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static float
doFma(int8_t fval, int8_t ival, float tmp) {
    (void)fval;
    (void)ival;
    (void)tmp;
    return 0;
}

#define checkCudaErr(...)                                                            \
    do {                                                                             \
        int64_t err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        REQUIRE(err == 0);                                                           \
    } while (0)

#define checkCudnnErr(...)                                                            \
    do {                                                                              \
        int64_t err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        REQUIRE(err == 0);                                                            \
    } while (0)

template <typename T_ELEM>
class SurfaceManager {
   public:
    T_ELEM* devPtrX         = NULL;
    T_ELEM* devPtrW         = NULL;
    T_ELEM* devPtrY         = NULL;
    T_ELEM* devPtrZ         = NULL;
    T_ELEM* devPtrB         = NULL;
    T_ELEM* devPtrAfterAdd  = NULL;
    T_ELEM* devPtrAfterConv = NULL;
    T_ELEM* devPtrAfterBias = NULL;

    T_ELEM* hostX         = NULL;
    T_ELEM* hostW         = NULL;
    T_ELEM* hostY         = NULL;
    T_ELEM* hostZ         = NULL;
    T_ELEM* hostB         = NULL;
    T_ELEM* hostAfterAdd  = NULL;
    T_ELEM* hostAfterConv = NULL;
    T_ELEM* hostAfterBias = NULL;
    T_ELEM* host_ref      = NULL;

    explicit SurfaceManager(int64_t Xsize, int64_t Wsize, int64_t Ysize, int64_t ref_size) try {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
        /*
        DPCT1064:832: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtrX = (T_ELEM*)sycl::malloc_device(size_t((Xsize) * sizeof(devPtrX[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:833: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtrW = (T_ELEM*)sycl::malloc_device(size_t((Wsize) * sizeof(devPtrW[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:834: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtrY = (T_ELEM*)sycl::malloc_device(size_t((Ysize) * sizeof(devPtrY[0])), dpct::get_in_order_queue())));

        hostX    = (T_ELEM*)calloc(size_t(Xsize), sizeof(hostX[0]));
        hostW    = (T_ELEM*)calloc(size_t(Wsize), sizeof(hostW[0]));
        hostY    = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostY[0]));
        host_ref = (T_ELEM*)calloc(size_t(ref_size), sizeof(host_ref[0]));

        initImage(hostX, Xsize);
        initImage(hostW, Wsize);
        initImage(hostY, Ysize);

        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtrX, hostX, size_t(sizeof(hostX[0]) * Xsize)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtrW, hostW, size_t(sizeof(hostW[0]) * Wsize)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtrY, hostY, size_t(sizeof(hostY[0]) * Ysize)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    explicit SurfaceManager(int64_t Xsize, int64_t Wsize, int64_t Ysize, int64_t Bsize, bool isConvBiasAdd) try {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
        (void)isConvBiasAdd;

        /*
        DPCT1064:835: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtrX = (T_ELEM*)sycl::malloc_device(size_t((Xsize) * sizeof(devPtrX[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:836: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtrW = (T_ELEM*)sycl::malloc_device(size_t((Wsize) * sizeof(devPtrW[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:837: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtrY = (T_ELEM*)sycl::malloc_device(size_t((Ysize) * sizeof(devPtrY[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:838: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtrZ = (T_ELEM*)sycl::malloc_device(size_t((Ysize) * sizeof(devPtrZ[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:839: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtrB = (T_ELEM*)sycl::malloc_device(size_t((Bsize) * sizeof(devPtrB[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:840: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(devPtrAfterConv = (T_ELEM*)sycl::malloc_device(
                                          size_t((Ysize) * sizeof(devPtrAfterConv[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:841: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(devPtrAfterAdd = (T_ELEM*)sycl::malloc_device(
                                          size_t((Ysize) * sizeof(devPtrAfterAdd[0])), dpct::get_in_order_queue())));
        /*
        DPCT1064:842: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(devPtrAfterBias = (T_ELEM*)sycl::malloc_device(
                                          size_t((Ysize) * sizeof(devPtrAfterBias[0])), dpct::get_in_order_queue())));

        hostX         = (T_ELEM*)calloc(size_t(Xsize), sizeof(hostX[0]));
        hostW         = (T_ELEM*)calloc(size_t(Wsize), sizeof(hostW[0]));
        hostY         = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostY[0]));
        hostZ         = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostZ[0]));
        hostB         = (T_ELEM*)calloc(size_t(Bsize), sizeof(hostB[0]));
        hostAfterConv = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostAfterConv[0]));
        hostAfterAdd  = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostAfterAdd[0]));
        hostAfterBias = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostAfterBias[0]));
        host_ref      = (T_ELEM*)calloc(size_t(Ysize), sizeof(host_ref[0]));

        initImage(hostX, Xsize);
        initImage(hostW, Wsize);
        initImage(hostY, Ysize);
        initImage(hostZ, Ysize);
        initImage(hostB, Bsize);
        initImage(hostAfterAdd, Ysize);
        initImage(hostAfterBias, Ysize);
        initImage(hostAfterConv, Ysize);

        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtrX, hostX, (size_t)(sizeof(hostX[0]) * Xsize)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtrW, hostW, (size_t)(sizeof(hostW[0]) * Wsize)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtrY, hostY, (size_t)(sizeof(hostY[0]) * Ysize)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtrZ, hostZ, (size_t)(sizeof(hostZ[0]) * Ysize)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtrB, hostB, (size_t)(sizeof(hostB[0]) * Bsize)).wait()));
        checkCudaErr(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(devPtrAfterAdd, hostAfterAdd, (size_t)(sizeof(hostAfterAdd[0]) * Ysize))
                                 .wait()));
        checkCudaErr(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(devPtrAfterBias, hostAfterBias, (size_t)(sizeof(hostAfterBias[0]) * Ysize))
                                 .wait()));
        checkCudaErr(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(devPtrAfterConv, hostAfterConv, (size_t)(sizeof(hostAfterConv[0]) * Ysize))
                                 .wait()));

        checkCudaErr(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    ~SurfaceManager() {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
        if (devPtrX) dpct::dpct_free(devPtrX, q_ct1);
        if (devPtrW) dpct::dpct_free(devPtrW, q_ct1);
        if (devPtrY) dpct::dpct_free(devPtrY, q_ct1);
        if (devPtrZ) dpct::dpct_free(devPtrZ, q_ct1);
        if (devPtrB) dpct::dpct_free(devPtrB, q_ct1);
        if (devPtrAfterAdd) dpct::dpct_free(devPtrAfterAdd, q_ct1);
        if (devPtrAfterBias) dpct::dpct_free(devPtrAfterBias, q_ct1);
        if (devPtrAfterConv) dpct::dpct_free(devPtrAfterConv, q_ct1);

        if (hostX) free(hostX);
        if (hostW) free(hostW);
        if (hostY) free(hostY);
        if (hostZ) free(hostZ);
        if (hostB) free(hostB);
        if (hostAfterAdd) free(hostAfterAdd);
        if (hostAfterBias) free(hostAfterBias);
        if (hostAfterConv) free(hostAfterConv);
        if (host_ref) free(host_ref);
    }
};

template <typename T_ELEM>
struct Surface {
    T_ELEM* devPtr     = NULL;
    T_ELEM* hostPtr    = NULL;
    T_ELEM* hostRefPtr = NULL;
    int64_t n_elems    = 0;

    explicit Surface(int64_t n_elems, bool hasRef) try : n_elems(n_elems) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
        /*
        DPCT1064:843: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(devPtr = (T_ELEM*)sycl::malloc_device((size_t)((n_elems) * sizeof(devPtr[0])),
                                                                            dpct::get_in_order_queue())));
        hostPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostPtr[0]));
        if (hasRef) {
            hostRefPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostRefPtr[0]));
        }
        initImage(hostPtr, n_elems);
        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    explicit Surface(int64_t n_elems, bool hasRef, bool isInterleaved) try {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
        (void)isInterleaved;
        /*
        DPCT1064:844: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtr = (T_ELEM*)sycl::malloc_device((n_elems) * sizeof(devPtr[0]), dpct::get_in_order_queue())));
        hostPtr = (T_ELEM*)calloc(n_elems, sizeof(hostPtr[0]));
        if (hasRef) {
            hostRefPtr = (T_ELEM*)calloc(n_elems, sizeof(hostRefPtr[0]));
        }
        initImage(hostPtr, n_elems);
        uint32_t* temp = (uint32_t*)hostPtr;
        for (auto i = 0; i < n_elems; i = i + 2) {
            temp[i + 1] = 1u;
        }

        checkCudaErr(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems)).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    explicit Surface(int64_t size, bool hasRef, T_ELEM fillValue) try : n_elems(size) {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1        = dev_ct1.in_order_queue();
        /*
        DPCT1064:845: Migrated cudaMalloc call is used in a macro/template definition and may not be valid for all
        macro/template uses. Adjust the code.
        */
        checkCudaErr(DPCT_CHECK_ERROR(
            devPtr = (T_ELEM*)sycl::malloc_device((size) * sizeof(devPtr[0]), dpct::get_in_order_queue())));
        hostPtr = (T_ELEM*)calloc(size, sizeof(hostPtr[0]));
        if (hasRef) {
            hostRefPtr = (T_ELEM*)calloc(n_elems, sizeof(hostRefPtr[0]));
        }
        for (int i = 0; i < size; i++) {
            hostPtr[i] = fillValue;
        }
        checkCudaErr(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems).wait()));
        checkCudaErr(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    ~Surface() {
        if (devPtr) {
            dpct::dpct_free(devPtr, dpct::get_in_order_queue());
            devPtr = nullptr;
        }
        if (hostPtr) {
            free(hostPtr);
            hostPtr = nullptr;
        }
        if (hostRefPtr) {
            free(hostRefPtr);
            hostRefPtr = nullptr;
        }
    }
};
