#include <dpct/dnnl_utils.hpp>
/*
 * Copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "cudnn_frontend.h"

#if (CUDNN_VERSION >= 8600)

#include "resnet_sample.h"

void
RunResidualBlock(cudnn_frontend::ResidualBlockParams const &params,
                 cudnn_frontend::ResidualBlockDevPtrStore *devPtrStore,
                 const std::string &type) try {
    if (check_device_arch_newer_than("hopper") == false) {
        return;
    }
    dpct::dnnl::engine_ext handle;
    checkCudnnErr(DPCT_CHECK_ERROR(handle.create_engine()));

    // Instantiate a shared ptr to a residual block
    std::shared_ptr<cudnn_frontend::IBlock> residualBlock;

    // Creates Residual Block with params
    dpct::err1 status = cudnn_frontend::createResidualBlock(handle, type, residualBlock, params);

    if (status != 0) {
        if (residualBlock == nullptr && status == 3000) {
            std::cout << "[ERROR]: Block type not supported" << std::endl;
            CHECK(false);
        }
        std::cout << residualBlock->getErrorMessage() << std::endl;
        CHECK(false);
    }

    auto workspace_size = residualBlock->getWorkspaceSize();
    std::cout << "Residual block " << type << " requires workspace " << workspace_size << " bytes." << std::endl;

    void *workspace_ptr = NULL;
    checkCudaErr(
        DPCT_CHECK_ERROR(workspace_ptr = (void *)sycl::malloc_device(workspace_size, dpct::get_in_order_queue())));

    status = cudnn_frontend::setWorkspace(devPtrStore, type, workspace_ptr);
    if (status != 0) {
        CHECK(false);
    }

    // Creates variant packs based on devPtrStore and executes
    status = cudnn_frontend::runBlock(handle, residualBlock, devPtrStore);

    if (status != 0) {
        std::cout << residualBlock->getErrorMessage() << std::endl;
        CHECK(false);
    }
    dpct::get_current_device().queues_wait_and_throw();
    /*
    DPCT1027:1208: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
    */
    checkCudnnErr(0);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif
