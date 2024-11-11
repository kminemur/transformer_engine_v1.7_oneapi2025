#pragma once
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <string>
#include "layers/common/include/cudnn_frontend_blocks.h"

#ifdef WIN32
#define strncasecmp strnicmp
#endif

namespace cudnn_frontend {

/**
 * @brief Create a Residual Block object. Takes in a block type in the form of a string. Choose between "forward" and
 * "backward" to specify which residual block you want. Your configured params and device poitner store will handle the
 * rest of the creation. Returns a cudnnStatus_t denoting whether the block was created successfully
 *
 * @param handle a cudnnHandle for the residual block
 * @param blockType choose between "forward" and "backward"
 * @param blockPtr a std::shared_ptr<IBlock> reference to an IBlock shared ptr (You should be creating this)
 * @param params A ResidualBlockParams object to configure the block
 * @return cudnnStatus_t Returns CUDNN_STATUS_SUCCESS if block was successfully created, otherwise, if
 * NV_CUDNN_DISABLE_EXCEPTION is not defined, it returns a bad status with an error message. Otherwise, it throws an
 * error for which tensor failed to build. Use getErrorMessage() to see error message.
 */
static inline dpct::err1
createResidualBlock(dpct::dnnl::engine_ext &handle,
                    const std::string &blockType,
                    std::shared_ptr<IBlock> &blockPtr,
                    ResidualBlockParams const &params)

{
    if (strncasecmp(blockType.c_str(), "forward", blockType.size()) == 0) {
        blockPtr = std::shared_ptr<IBlock>(new ResidualForwardBlock(handle, params));
        if (blockPtr->getStatus() != 0) return blockPtr->getStatus();
        return blockPtr->build(handle);
    } else if (strncasecmp(blockType.c_str(), "forward_inference", blockType.size()) == 0) {
        blockPtr = std::shared_ptr<IBlock>(new ResidualForwardInferenceBlock(handle, params));
        if (blockPtr->getStatus() != 0) return blockPtr->getStatus();
        return blockPtr->build(handle);
    }

    return 3000;
}

template <typename DevPtrStoreType>
static inline dpct::err1
setWorkspace(DevPtrStoreType *devPtrStore, const std::string &blockType, void *workspace) {
    return devPtrStore->setWorkspace(blockType, workspace);
}

/**
 * @brief Runs the block by creating the necessary variant packs and executing the execution plans. All done internally
 * for the user. Returns a cudnnStatus_t if successful.
 *
 * @tparam DevPtrStoreType type of device pointer store (stem, residual, or classifier)
 * @param handle A cudnnHandle for the block (should be the same one you used to create the block)
 * @param blockPtr a std::shared_ptr<IBlock> reference to an IBlock shared ptr, should be the same ptr used when
 * creating the block
 * @param devPtrStore A DevPtrStore object which contains the necessary pointers to the device memory (see
 * cudnn_frontend_classifier_block_dev_ptr_store.h for details).
 * @return cudnnStatus_t Returns CUDNN_STATUS_SUCCESS if block was successfully created, otherwise, if
 * NV_CUDNN_DISABLE_EXCEPTION is not defined, it returns a bad status with an error message. Otherwise, it throws an
 * error for which tensor failed to build. Use getErrorMessage() to see error message.
 */
template <typename DevPtrStoreType>
static inline dpct::err1
runBlock(dpct::dnnl::engine_ext &handle, std::shared_ptr<IBlock> &block, DevPtrStoreType *devPtrStore) {
    block->setWorkspace(devPtrStore->getWorkspace(block->getDirection()));
    dpct::err1 status = block->createVariantPacks(devPtrStore);

    if (status != 0) {
        std::cout << block->getErrorMessage() << std::endl;
        return status;
    }
    return block->execute(handle);
}

}  // namespace cudnn_frontend

#undef strncasecmp