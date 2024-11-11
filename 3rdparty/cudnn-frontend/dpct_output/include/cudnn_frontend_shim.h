/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include <ostream>
#include <iostream>
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <dlfcn.h>
#include <mutex>
#endif

namespace cudnn_frontend {

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING

// cudnn package initialization set this global handle
extern void *cudnn_dlhandle;

inline void *
get_symbol(const char *function_name) {
    void *ret = dlsym(cudnn_dlhandle, function_name);
    return ret;
}

inline void *
get_cuda_symbol(const char *function_name) {
    static std::mutex cuda_fe_lib_mutex;
    std::lock_guard<std::mutex> lock(cuda_fe_lib_mutex);
    char *c                = NULL;
    c                      = dlerror();
    static void *dl_handle = dlopen("libcudart.so", RTLD_NOW);
    c                      = dlerror();
    (void)c;
    if (dl_handle == nullptr) {
        std::string error_msg = std::string("Unable to dlopen libcudart.so") + std::string(c);
        throw std::runtime_error(error_msg.c_str());
    }

    void *ret = dlsym(dl_handle, function_name);
    return ret;
}
#endif

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#define NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(MINIMUM_VERSION, DESCRIPTOR, MESSAGE) \
    if (MINIMUM_VERSION > get_backend_version()) {                                         \
        set_error_and_throw_exception(&DESCRIPTOR, CUDNN_STATUS_INVALID_VALUE, MESSAGE);   \
        return std::move(DESCRIPTOR);                                                      \
    }
#else
#define NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(MINIMUM_VERSION, DESCRIPTOR, MESSAGE)
#endif

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#define NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(MINIMUM_VERSION, STATUS) \
    if (MINIMUM_VERSION > get_backend_version()) {                               \
        return STATUS;                                                           \
    }
#else
#define NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(MINIMUM_VERSION, STATUS)
#endif

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#define NV_FE_CALL_TO_BACKEND(function_name, backend_symbol, ...)           \
    static void *fptr = get_symbol(#backend_symbol);                        \
    if (fptr == nullptr) {                                                  \
        throw std::runtime_error("Unable to find symbol " #backend_symbol); \
    }                                                                       \
    return reinterpret_cast<decltype(function_name) *>(fptr)(__VA_ARGS__);
#else
#define NV_FE_CALL_TO_BACKEND(function_name, backend_symbol, ...) return backend_symbol(__VA_ARGS__);
#endif

#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#define NV_FE_CALL_TO_CUDA(function_name, cuda_symbol, ...)              \
    static void *fptr = get_cuda_symbol(#cuda_symbol);                   \
    if (fptr == nullptr) {                                               \
        throw std::runtime_error("Unable to find symbol " #cuda_symbol); \
    }                                                                    \
    return reinterpret_cast<decltype(function_name) *>(fptr)(__VA_ARGS__);
#else
#define NV_FE_CALL_TO_CUDA(function_name, cuda_symbol, ...) return cuda_symbol(__VA_ARGS__)));
#endif

inline dpct::err0
cuda_event_create(dpct::event_ptr *event) try {
    NV_FE_CALL_TO_CUDA(cuda_event_create, DPCT_CHECK_ERROR(* = new sycl::event()), event);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err0
cuda_event_destroy(dpct::event_ptr event) try {
    NV_FE_CALL_TO_CUDA(cuda_event_destroy, DPCT_CHECK_ERROR(dpct::destroy_event(event)), event);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err0
cuda_event_record(dpct::event_ptr event, dpct::queue_ptr stream) try {
    /*
    DPCT1024:0: The original code returned the error code that was further consumed by the program logic. This original
    code was replaced with 0. You may need to rewrite the program logic consuming the error code.
    */
    NV_FE_CALL_TO_CUDA(
        cuda_event_record, DPCT_CHECK_ERROR(*event = stream->ext_oneapi_submit_barrier()), event, stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err0
cuda_event_synchronize(dpct::event_ptr event) try {
    NV_FE_CALL_TO_CUDA(cuda_event_synchronize, DPCT_CHECK_ERROR(event->wait_and_throw()), event);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err0
cuda_event_elapsed_time(float *ms, dpct::event_ptr start, dpct::event_ptr end) try {
    NV_FE_CALL_TO_CUDA(
        cuda_event_elapsed_time,
        DPCT_CHECK_ERROR(*(ms) = (end->get_profiling_info<sycl::info::event_profiling::command_end>() -
                                  start->get_profiling_info<sycl::info::event_profiling::command_start>()) /
                                 1000000.0f),
        ms,
        start,
        end);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err0
cuda_mem_cpy_async(void *dst, const void *src, size_t count, dpct::memcpy_direction kind, dpct::queue_ptr stream) try {
    /*
    DPCT1124:1: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the origin API might be synchronous, it
    depends on the type of operand memory, so you may need to call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    NV_FE_CALL_TO_CUDA(cuda_mem_cpy_async, DPCT_CHECK_ERROR(stream->memcpy, dst, src, count);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err0
cuda_mem_set_async(void *devPtr, int value, size_t count, dpct::queue_ptr stream) try {
    NV_FE_CALL_TO_CUDA(cuda_mem_set_async, DPCT_CHECK_ERROR(stream->memset, devPtr, value, count);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err0
cuda_get_device_properties(dpct::device_info *prop, int device) try {
    NV_FE_CALL_TO_CUDA(cuda_get_device_properties, cudaGetDeviceProperties, prop, device);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err0
cuda_get_device(int *device) try {
    NV_FE_CALL_TO_CUDA(
        cuda_get_device, DPCT_CHECK_ERROR(*device = dpct::dev_mgr::instance().current_device_id()), device);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline const char *
cuda_get_error_string(dpct::err0 error) {
    /*
    DPCT1009:2: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced by a
    placeholder string. You need to rewrite this code.
    */
    NV_FE_CALL_TO_CUDA(cuda_get_error_string, "<Placeholder string>", error);
}

inline dpct::err0
cuda_device_synchronize() try {
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    static void *fptr = get_cuda_symbol("cudaDeviceSynchronize");
    if (fptr == nullptr) {
        throw std::runtime_error("Unable to find symbol cudaDeviceSynchronize");
    }
    return reinterpret_cast<decltype(cuda_device_synchronize) *>(fptr)();
#else
    return DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline dpct::err1
create_handle(dpct::dnnl::engine_ext *handle) {
    NV_FE_CALL_TO_BACKEND(create_handle, DPCT_CHECK_ERROR(handle->create_engine()), handle);
}

inline dpct::err1
destroy_handle(dpct::dnnl::engine_ext handle) {
    /*
    DPCT1027:3: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
    */
    NV_FE_CALL_TO_BACKEND(destroy_handle, 0, handle);
}

inline size_t
get_backend_version(void) {
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
    static void *fptr = get_symbol("cudnnGetVersion");
    if (fptr == nullptr) {
        throw std::runtime_error("Unable to find symbol cudnnGetVersion");
    }
    return reinterpret_cast<decltype(get_backend_version) *>(fptr)();
#else
    return dpct::dnnl::get_version();
#endif
}

namespace detail {

inline std::string
convert_version_to_str(size_t const version) {
    // The multiplier for major version pre-v9 and post-v9 are different.
    size_t major = version / 10000;
    size_t minor = (version / 100) % 100;
    if (major == 0) {
        major = version / 1000;
        minor = (version / 100) % 10;
    }
    auto patch = version % 100;

    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}
}  // namespace detail

inline std::string
get_backend_version_string() {
    return detail::convert_version_to_str(get_backend_version());
}

inline dpct::err1
create_descriptor(cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor) {
    /*
    DPCT1007:4: Migration of cudnnBackendCreateDescriptor is not supported.
    */
    NV_FE_CALL_TO_BACKEND(create_descriptor, cudnnBackendCreateDescriptor, descriptorType, descriptor);
}

inline dpct::err1
destroy_descriptor(cudnnBackendDescriptor_t descriptor) {
    /*
    DPCT1007:5: Migration of cudnnBackendDestroyDescriptor is not supported.
    */
    NV_FE_CALL_TO_BACKEND(destroy_descriptor, cudnnBackendDestroyDescriptor, descriptor);
}

inline dpct::err1
set_attribute(cudnnBackendDescriptor_t descriptor,
              cudnnBackendAttributeName_t attributeName,
              cudnnBackendAttributeType_t attributeType,
              int64_t elementCount,
              const void *arrayOfElements) {
    /*
    DPCT1007:6: Migration of cudnnBackendSetAttribute is not supported.
    */
    NV_FE_CALL_TO_BACKEND(set_attribute,
                          cudnnBackendSetAttribute,
                          descriptor,
                          attributeName,
                          attributeType,
                          elementCount,
                          arrayOfElements);
}

inline dpct::err1
get_attribute(cudnnBackendDescriptor_t const descriptor,
              cudnnBackendAttributeName_t attributeName,
              cudnnBackendAttributeType_t attributeType,
              int64_t requestedElementCount,
              int64_t *elementCount,
              void *arrayOfElements) {
    /*
    DPCT1007:7: Migration of cudnnBackendGetAttribute is not supported.
    */
    NV_FE_CALL_TO_BACKEND(get_attribute,
                          cudnnBackendGetAttribute,
                          descriptor,
                          attributeName,
                          attributeType,
                          requestedElementCount,
                          elementCount,
                          arrayOfElements)
}

inline dpct::err1
finalize(cudnnBackendDescriptor_t descriptor) {
    /*
    DPCT1007:8: Migration of cudnnBackendFinalize is not supported.
    */
    NV_FE_CALL_TO_BACKEND(finalize, cudnnBackendFinalize, descriptor);
}

inline dpct::err1
execute(dpct::dnnl::engine_ext handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t variantPack) {
    /*
    DPCT1007:9: Migration of cudnnBackendExecute is not supported.
    */
    NV_FE_CALL_TO_BACKEND(execute, cudnnBackendExecute, handle, executionPlan, variantPack);
}

inline const char *
get_error_string(dpct::err1 status) {
    /*
    DPCT1009:10: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced by a
    placeholder string. You need to rewrite this code.
    */
    NV_FE_CALL_TO_BACKEND(get_error_string, "<Placeholder string>", status);
}

inline dpct::err1
set_stream(dpct::dnnl::engine_ext handle, dpct::queue_ptr stream) {
    NV_FE_CALL_TO_BACKEND(set_stream, DPCT_CHECK_ERROR(.set_queue()), handle, stream);
}

inline dpct::err1
get_stream(dpct::dnnl::engine_ext handle, dpct::queue_ptr *stream) {
    NV_FE_CALL_TO_BACKEND(get_stream, DPCT_CHECK_ERROR(* =.get_queue()), handle, stream);
}

inline dpct::err1
create_filter_desc_v7(dpct::dnnl::memory_desc_ext *filter) {
    /*
    DPCT1027:11: The call to cudnnCreateFilterDescriptor was replaced with 0 because this functionality is redundant in
    SYCL.
    */
    NV_FE_CALL_TO_BACKEND(create_filter_desc_v7, 0, filter);
}

inline dpct::err1
set_ndfilter_desc_v7(dpct::dnnl::memory_desc_ext filter,
                     dpct::library_data_t type,
                     dpct::dnnl::memory_format_tag format,
                     int x,
                     const int filterDimA[]) {
    NV_FE_CALL_TO_BACKEND(set_ndfilter_desc_v7, DPCT_CHECK_ERROR(.set(, , , )), filter, type, format, x, filterDimA);
}

inline dpct::err1
reorder_filter_bias(dpct::dnnl::engine_ext handle,
                    const dpct::dnnl::memory_desc_ext filterDesc,
                    cudnnReorderType_t reorderType,
                    const void *filterData,
                    void *reorderedFilterData,
                    int reorderBias,
                    const void *biasData,
                    void *reorderedBiasData) {
    /*
    DPCT1007:12: Migration of cudnnReorderFilterAndBias is not supported.
    */
    NV_FE_CALL_TO_BACKEND(reorder_filter_bias,
                          cudnnReorderFilterAndBias,
                          handle,
                          filterDesc,
                          reorderType,
                          filterData,
                          reorderedFilterData,
                          reorderBias,
                          biasData,
                          reorderedBiasData);
}

inline dpct::err1
destroy_filter(dpct::dnnl::memory_desc_ext filter) {
    /*
    DPCT1027:13: The call to cudnnDestroyFilterDescriptor was replaced with 0 because this functionality is redundant in
    SYCL.
    */
    NV_FE_CALL_TO_BACKEND(destroy_filter, 0, filter);
}
}  // namespace cudnn_frontend
