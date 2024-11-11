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
#include <iterator>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include "cudnn_frontend_EngineConfig.h"
#include "cudnn_frontend_Engine.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// ExecutionPlan_v8 Class
/// This class tells the Configuration of the Engine in terms of the knob
/// choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine
///
/// Use ExecutionPlanBuilder_v8 to build this class.
/// Describe returns a string describing the tensor class
///
class ExecutionPlan_v8 : public BackendDescriptor {
   public:
    friend class ExecutionPlanBuilder_v8;

    ExecutionPlan_v8(ExecutionPlan_v8 &&from) = default;
    ExecutionPlan_v8 &
    operator=(ExecutionPlan_v8 &&) = default;

    ~ExecutionPlan_v8() = default;
    /** @defgroup ExecutionPlanQuery
     *  Query individual property of ExecutionPlan_v8 class
     *  @{
     */
    //! Query the workspace requirement for the given plan
    auto
    getWorkspaceSize(void) const -> int64_t {
        return workSpaceSize;
    }

    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR : ";
        ss << getTag() << ", ";
        ss << "numeric_notes:"
           << "[";
        for (auto note : numeric_notes_vec) {
            ss << cudnn_frontend::to_string(note) << ",";
        }
        ss << "] behavior_notes:"
           << "[";
        for (auto note : behavior_notes_vec) {
            ss << cudnn_frontend::to_string(note) << ",";
        }
        ss << "] workSpaceSize: " << workSpaceSize;
        return ss.str();
    }

    std::string const &
    getTag() const {
        return planTag;
    }

    void
    setExecutionTime(float time_) {
        execution_time_ms = time_;
    }

    float
    getExecutionTime() const {
        return execution_time_ms;
    }

    std::vector<cudnnBackendNumericalNote_t> const &
    getAllNumericNotes() const {
        return numeric_notes_vec;
    }

    /*
    DPCT1007:650: Migration of CUDNN_NUMERICAL_NOTE_TYPE_COUNT is not supported.
    */
    std::array<cudnnBackendNumericalNote_t, CUDNN_NUMERICAL_NOTE_TYPE_COUNT> const &
    getNumericNotes() const {
        return numeric_notes;
    }

    /*
    DPCT1007:651: Migration of CUDNN_BEHAVIOR_NOTE_TYPE_COUNT is not supported.
    */
    std::array<cudnnBackendBehaviorNote_t, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT> const &
    getBehaviorNotes() const {
        return behavior_notes;
    }
    std::vector<cudnnBackendBehaviorNote_t> const &
    getAllBehaviorNotes() const {
        return behavior_notes_vec;
    }

    std::string
    getJsonRepresentation() const {
        auto status = 0;
        int64_t serializationSize;
        std::vector<char> serialization_buf;
        status = cudnn_frontend::get_attribute(
            pointer->get_backend_descriptor(),
            /*
            DPCT1007:652: Migration of CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION is not supported.
            */
            CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
            /*
            DPCT1007:653: Migration of CUDNN_TYPE_CHAR is not supported.
            */
            CUDNN_TYPE_CHAR,
            0,
            &serializationSize,
            nullptr);
        if (status != 0) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION Failed");
        }
        serialization_buf.resize(static_cast<size_t>(serializationSize));
        status = cudnn_frontend::get_attribute(
            pointer->get_backend_descriptor(),
            /*
            DPCT1007:654: Migration of CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION is not supported.
            */
            CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
            /*
            DPCT1007:655: Migration of CUDNN_TYPE_CHAR is not supported.
            */
            CUDNN_TYPE_CHAR,
            serializationSize,
            &serializationSize,
            serialization_buf.data());
        if (status != 0) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION Failed");
        }
        std::string json_string(serialization_buf.begin(), serialization_buf.end());
        return json_string;
    }

    ExecutionPlan_v8(ExecutionPlan_v8 const &) = default;
    ExecutionPlan_v8 &
    operator=(ExecutionPlan_v8 const &) = default;

   private:
    void
    fetchNotes(ManagedOpaqueDescriptor &extractedEngine) {
        auto status                               = 0;
        int64_t elem_count                        = 0;
        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
        status =
            cudnn_frontend::get_attribute(extractedEngine_,
                                          /*
                                          DPCT1007:656: Migration of CUDNN_ATTR_ENGINE_NUMERICAL_NOTE is not supported.
                                          */
                                          CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                          /*
                                          DPCT1007:657: Migration of CUDNN_TYPE_NUMERICAL_NOTE is not supported.
                                          */
                                          CUDNN_TYPE_NUMERICAL_NOTE,
                                          /*
                                          DPCT1007:658: Migration of CUDNN_NUMERICAL_NOTE_TYPE_COUNT is not supported.
                                          */
                                          CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                          &elem_count,
                                          nullptr);
        numeric_notes_vec.resize(static_cast<size_t>(elem_count));
        status =
            cudnn_frontend::get_attribute(extractedEngine_,
                                          /*
                                          DPCT1007:659: Migration of CUDNN_ATTR_ENGINE_NUMERICAL_NOTE is not supported.
                                          */
                                          CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                          /*
                                          DPCT1007:660: Migration of CUDNN_TYPE_NUMERICAL_NOTE is not supported.
                                          */
                                          CUDNN_TYPE_NUMERICAL_NOTE,
                                          /*
                                          DPCT1007:661: Migration of CUDNN_NUMERICAL_NOTE_TYPE_COUNT is not supported.
                                          */
                                          CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                          &elem_count,
                                          numeric_notes_vec.data());
        ptrdiff_t end =
            /*
            DPCT1007:662: Migration of CUDNN_NUMERICAL_NOTE_TYPE_COUNT is not supported.
            */
            static_cast<ptrdiff_t>(std::min(elem_count, static_cast<int64_t>(CUDNN_NUMERICAL_NOTE_TYPE_COUNT)));
        std::copy(numeric_notes_vec.begin(), numeric_notes_vec.begin() + end, numeric_notes.begin());
        if (static_cast<size_t>(elem_count) < numeric_notes.size())
            std::fill_n(numeric_notes.begin() + static_cast<size_t>(elem_count),
                        numeric_notes.size() - static_cast<size_t>(elem_count),
                        /*
                        DPCT1007:663: Migration of CUDNN_NUMERICAL_NOTE_TYPE_COUNT is not supported.
                        */
                        CUDNN_NUMERICAL_NOTE_TYPE_COUNT);
        if (status != 0) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINE_NUMERICAL_NOTE Failed");
        }
        status =
            cudnn_frontend::get_attribute(extractedEngine_,
                                          /*
                                          DPCT1007:664: Migration of CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE is not supported.
                                          */
                                          CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                          /*
                                          DPCT1007:665: Migration of CUDNN_TYPE_BEHAVIOR_NOTE is not supported.
                                          */
                                          CUDNN_TYPE_BEHAVIOR_NOTE,
                                          /*
                                          DPCT1007:666: Migration of CUDNN_BEHAVIOR_NOTE_TYPE_COUNT is not supported.
                                          */
                                          CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                          &elem_count,
                                          nullptr);
        behavior_notes_vec.resize(static_cast<size_t>(elem_count));
        status =
            cudnn_frontend::get_attribute(extractedEngine_,
                                          /*
                                          DPCT1007:667: Migration of CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE is not supported.
                                          */
                                          CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                          /*
                                          DPCT1007:668: Migration of CUDNN_TYPE_BEHAVIOR_NOTE is not supported.
                                          */
                                          CUDNN_TYPE_BEHAVIOR_NOTE,
                                          /*
                                          DPCT1007:669: Migration of CUDNN_BEHAVIOR_NOTE_TYPE_COUNT is not supported.
                                          */
                                          CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
                                          &elem_count,
                                          behavior_notes_vec.data());
        /*
        DPCT1007:670: Migration of CUDNN_BEHAVIOR_NOTE_TYPE_COUNT is not supported.
        */
        end = static_cast<ptrdiff_t>(std::min(elem_count, static_cast<int64_t>(CUDNN_BEHAVIOR_NOTE_TYPE_COUNT)));
        std::copy(behavior_notes_vec.begin(), behavior_notes_vec.begin() + end, behavior_notes.begin());
        if (static_cast<size_t>(elem_count) < behavior_notes.size())
            std::fill_n(behavior_notes.begin() + static_cast<size_t>(elem_count),
                        behavior_notes.size() - static_cast<size_t>(elem_count),
                        /*
                        DPCT1007:671: Migration of CUDNN_BEHAVIOR_NOTE_TYPE_COUNT is not supported.
                        */
                        CUDNN_BEHAVIOR_NOTE_TYPE_COUNT);
        if (status != 0) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE Failed");
        }
    }

    void
    buildTag(ManagedOpaqueDescriptor &extractedEngine) {
        // Compute a unique tag for execution plan:
        auto status = 0;
        std::stringstream tag{""};
        int64_t elemCount = 0, engineId = 0, numKnobs = 0;

        /*
        DPCT1007:672: Migration of CUDNN_KNOB_TYPE_COUNTS is not supported.
        */
        std::array<ManagedOpaqueDescriptor, CUDNN_KNOB_TYPE_COUNTS> extractedKnobs{{nullptr}};
        for (auto &knob : extractedKnobs) {
            /*
            DPCT1007:673: Migration of CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR is not supported.
            */
            knob   = make_shared_backend_pointer(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
            status = knob->get_status();
            if (status != 0) {
                set_error_and_throw_exception(
                    this, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed when compute tag");
            }
        }

        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
        /*
        DPCT1007:674: Migration of CUDNN_KNOB_TYPE_COUNTS is not supported.
        */
        std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> extractedKnobs_{{nullptr}};
        for (std::uint32_t i = 0; i < extractedKnobs.size(); i++) {
            extractedKnobs_[i] = extractedKnobs[i]->get_backend_descriptor();
        }

        status = cudnn_frontend::get_attribute(
            /*
            DPCT1007:675: Migration of CUDNN_ATTR_ENGINE_GLOBAL_INDEX is not supported.
            */
            /*
            DPCT1007:676: Migration of CUDNN_TYPE_INT64 is not supported.
            */
            extractedEngine_,
            CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
            CUDNN_TYPE_INT64,
            1,
            &elemCount,
            &engineId);
        if (status != 0) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINE_GLOBAL_INDEX Failed");
        }
        tag << "eng" << engineId;

        status =
            cudnn_frontend::get_attribute(engine_config->get_backend_descriptor(),
                                          /*
                                          DPCT1007:677: Migration of CUDNN_ATTR_ENGINECFG_KNOB_CHOICES is not supported.
                                          */
                                          CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                          /*
                                          DPCT1007:678: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          /*
                                          DPCT1007:679: Migration of CUDNN_KNOB_TYPE_COUNTS is not supported.
                                          */
                                          CUDNN_KNOB_TYPE_COUNTS,
                                          &numKnobs,
                                          &(extractedKnobs_[0]));
        if (status != 0) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_KNOB_CHOICES Failed");
        }
        /*
        DPCT1007:680: Migration of CUDNN_KNOB_TYPE_COUNTS is not supported.
        */
        if (numKnobs > CUDNN_KNOB_TYPE_COUNTS) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "numKnobs exceed the CUDNN_KNOB_TYPE_COUNTS");
        }
        for (size_t idx = 0; idx < static_cast<size_t>(numKnobs); ++idx) {
            const cudnnBackendDescriptor_t &knob = extractedKnobs_[idx];
            /*
            DPCT1007:681: Migration of CUDNN_KNOB_TYPE_COUNTS is not supported.
            */
            cudnnBackendKnobType_t type          = CUDNN_KNOB_TYPE_COUNTS;
            int64_t choice                       = -2;
            status                               = cudnn_frontend::get_attribute(
                /*
                DPCT1007:682: Migration of CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE is not supported.
                */
                /*
                DPCT1007:683: Migration of CUDNN_TYPE_KNOB_TYPE is not supported.
                */
                knob,
                CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,
                CUDNN_TYPE_KNOB_TYPE,
                1,
                nullptr,
                &type);
            if (status != 0) {
                set_error_and_throw_exception(this,
                                              status,
                                              "computeTag CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE Failed");
            }
            status = cudnn_frontend::get_attribute(
                /*
                DPCT1007:684: Migration of CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE is not supported.
                */
                /*
                DPCT1007:685: Migration of CUDNN_TYPE_INT64 is not supported.
                */
                knob,
                CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,
                CUDNN_TYPE_INT64,
                1,
                nullptr,
                &choice);
            if (status != 0) {
                set_error_and_throw_exception(this,
                                              status,
                                              "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                              "CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE Failed");
            }
            tag << "_k" << type << "=" << choice;
        }
        planTag += tag.str();
    }

    void
    computeWorkSpaceSize() {
        auto status = cudnn_frontend::get_attribute(
            pointer->get_backend_descriptor(),
            /*
            DPCT1007:686: Migration of CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE is not supported.
            */
            CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
            /*
            DPCT1007:687: Migration of CUDNN_TYPE_INT64 is not supported.
            */
            CUDNN_TYPE_INT64,
            1,
            nullptr,
            &workSpaceSize);
        if (status != 0) {
            set_error_and_throw_exception(this,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE Failed");
        }
        if (workSpaceSize < 0) {
            set_error_and_throw_exception(
                this, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute Workspace Size Invalid");
        }
    }

    ExecutionPlan_v8()                    = default;
    ManagedOpaqueDescriptor engine_config = nullptr;
    dpct::dnnl::engine_ext handle         = nullptr;
    std::string planTag;

    std::int64_t workSpaceSize = 0;
    /*
    DPCT1007:688: Migration of CUDNN_NUMERICAL_NOTE_TYPE_COUNT is not supported.
    */
    std::array<cudnnBackendNumericalNote_t, CUDNN_NUMERICAL_NOTE_TYPE_COUNT> numeric_notes;
    std::vector<cudnnBackendNumericalNote_t> numeric_notes_vec;
    /*
    DPCT1007:689: Migration of CUDNN_BEHAVIOR_NOTE_TYPE_COUNT is not supported.
    */
    std::array<cudnnBackendBehaviorNote_t, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT> behavior_notes;
    std::vector<cudnnBackendBehaviorNote_t> behavior_notes_vec;

    float execution_time_ms = 0.0f;
};

///
/// ExecutionPlanBuilder_v8 Class
/// Helper class used to build ExecutionPlan_v8 class
class ExecutionPlanBuilder_v8 {
   public:
    /** @defgroup ExecutionPlanBuilder_v8
     *  Set individual property of ExecutionPlan_v8 class
     *  @{
     */
    //! Set engine for the ExecutionPlan_v8
    auto
    setHandle(dpct::dnnl::engine_ext handle_) -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.handle = handle_;
        return *this;
    }
    //! Set engine Config for the Plan
    auto
    setEngineConfig(EngineConfig_v8 const &engine_config_) -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.engine_config = engine_config_.get_desc();
        m_execution_plan.planTag       = engine_config_.getTag();
        return *this;
    }

    //! Set engine Config for the Plan
    auto
    setEngineConfig(ManagedOpaqueDescriptor &desc, std::string const &opGraphTag_ = "") -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.engine_config = desc;
        m_execution_plan.planTag       = opGraphTag_;
        return *this;
    }

    auto
    setEngineConfig(ManagedOpaqueDescriptor const &desc, std::string const &opGraphTag_ = "")
        -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.engine_config = desc;
        m_execution_plan.planTag       = opGraphTag_;
        return *this;
    }
    /** @} */

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    ExecutionPlan_v8 &&
    build() {
        if (m_execution_plan.handle == nullptr || !m_execution_plan.handle) {
            set_error_and_throw_exception(
                &m_execution_plan,
                2000,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set the CUDNN_ATTR_EXECUTION_PLAN_HANDLE");
            return std::move(m_execution_plan);
        };
        if (m_execution_plan.engine_config == nullptr) {
            set_error_and_throw_exception(
                &m_execution_plan,
                2000,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set the CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG");
            return std::move(m_execution_plan);
        };

        // Create a descriptor. Memory allocation happens here.
        /*
        DPCT1007:690: Migration of CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR is not supported.
        */
        auto status = m_execution_plan.initialize_managed_backend_pointer(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_execution_plan);
        }

        status = cudnn_frontend::set_attribute(
            m_execution_plan.pointer->get_backend_descriptor(),
            /*
            DPCT1007:691: Migration of CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG is not supported.
            */
            CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            /*
            DPCT1007:692: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &(m_execution_plan.engine_config->get_backend_descriptor()));
        if (status != 0) {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG Failed");
            return std::move(m_execution_plan);
        }
        status =
            cudnn_frontend::set_attribute(m_execution_plan.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:693: Migration of CUDNN_ATTR_EXECUTION_PLAN_HANDLE is not supported.
                                          */
                                          CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                          /*
                                          DPCT1007:694: Migration of CUDNN_TYPE_HANDLE is not supported.
                                          */
                                          CUDNN_TYPE_HANDLE,
                                          1,
                                          &m_execution_plan.handle);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute CUDNN_ATTR_EXECUTION_PLAN_HANDLE Failed");
            return std::move(m_execution_plan);
        }
        // Finalizing the descriptor
        status = cudnn_frontend::finalize(m_execution_plan.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(
                &m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed");
            return std::move(m_execution_plan);
        }

        /*
        DPCT1007:695: Migration of CUDNN_BACKEND_ENGINE_DESCRIPTOR is not supported.
        */
        ManagedOpaqueDescriptor extractedEngine = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        status                                  = extractedEngine->get_status();
        if (status != 0) {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate of "
                                          "CUDNN_BACKEND_ENGINE_DESCRIPTOR failed when compute tag");
            return std::move(m_execution_plan);
        }
        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();
        int64_t elemCount                         = 0;
        status =
            cudnn_frontend::get_attribute(m_execution_plan.engine_config->get_backend_descriptor(),
                                          /*
                                          DPCT1007:696: Migration of CUDNN_ATTR_ENGINECFG_ENGINE is not supported.
                                          */
                                          CUDNN_ATTR_ENGINECFG_ENGINE,
                                          /*
                                          DPCT1007:697: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &elemCount,
                                          &extractedEngine_);
        if (status != 0) {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_ENGINE Failed");
            return std::move(m_execution_plan);
        }

        m_execution_plan.buildTag(extractedEngine);
        m_execution_plan.fetchNotes(extractedEngine);
        m_execution_plan.computeWorkSpaceSize();

        getLogger() << "[cudnn_frontend] " << m_execution_plan << std::endl;
        return std::move(m_execution_plan);
    }

    ExecutionPlan_v8 &&
    loadFromJson(const std::string &json_plan) {
        CUDNN_FRONTEND_UNUSED(json_plan);
        auto status = 0;

        if (m_execution_plan.handle == nullptr) {
            set_error_and_throw_exception(
                &m_execution_plan,
                2000,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: Check and Set the CUDNN_ATTR_EXECUTION_PLAN_HANDLE");
            return std::move(m_execution_plan);
        };

        // Create a descriptor. Memory allocation happens here.
        /*
        DPCT1007:698: Migration of CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR is not supported.
        */
        status = m_execution_plan.initialize_managed_backend_pointer(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_execution_plan);
        }

        std::vector<char> serialization_buf;
        serialization_buf.assign(json_plan.begin(), json_plan.end());
        status = cudnn_frontend::set_attribute(
            m_execution_plan.pointer->get_backend_descriptor(),
            /*
            DPCT1007:699: Migration of CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION is not supported.
            */
            CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
            /*
            DPCT1007:700: Migration of CUDNN_TYPE_CHAR is not supported.
            */
            CUDNN_TYPE_CHAR,
            serialization_buf.size(),
            serialization_buf.data());
        if (status != 0) {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION Failed");
            return std::move(m_execution_plan);
        }

        status =
            cudnn_frontend::set_attribute(m_execution_plan.pointer->get_backend_descriptor(),
                                          /*
                                          DPCT1007:701: Migration of CUDNN_ATTR_EXECUTION_PLAN_HANDLE is not supported.
                                          */
                                          CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                          /*
                                          DPCT1007:702: Migration of CUDNN_TYPE_HANDLE is not supported.
                                          */
                                          CUDNN_TYPE_HANDLE,
                                          1,
                                          &m_execution_plan.handle);
        if (status != 0) {
            set_error_and_throw_exception(
                &m_execution_plan,
                status,
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: SetAttribute CUDNN_ATTR_EXECUTION_PLAN_HANDLE Failed");
            return std::move(m_execution_plan);
        }

        status = cudnn_frontend::finalize(m_execution_plan.pointer->get_backend_descriptor());
        if (status != 0) {
            set_error_and_throw_exception(
                &m_execution_plan, status, "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed");
            return std::move(m_execution_plan);
        }

        /*
        DPCT1007:703: Migration of CUDNN_BACKEND_ENGINECFG_DESCRIPTOR is not supported.
        */
        m_execution_plan.engine_config = make_shared_backend_pointer(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        status                         = m_execution_plan.engine_config->get_status();
        if (status != 0) {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate of "
                                          "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR failed when computing tag");
            return std::move(m_execution_plan);
        }

        cudnnBackendDescriptor_t engCfgDesc = m_execution_plan.engine_config->get_backend_descriptor();
        int64_t elemCount                   = 0;
        status                              = cudnn_frontend::get_attribute(
            m_execution_plan.pointer->get_backend_descriptor(),
            /*
            DPCT1007:704: Migration of CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG is not supported.
            */
            CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            /*
            DPCT1007:705: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
            */
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1,
            &elemCount,
            &engCfgDesc);

        if (status != 0) {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG Failed");
            return std::move(m_execution_plan);
        }
        /*
        DPCT1007:706: Migration of CUDNN_BACKEND_ENGINE_DESCRIPTOR is not supported.
        */
        ManagedOpaqueDescriptor extractedEngine = make_shared_backend_pointer(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        status                                  = extractedEngine->get_status();
        if (status != 0) {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnCreate of "
                                          "CUDNN_BACKEND_ENGINE_DESCRIPTOR failed when computing tag");
            return std::move(m_execution_plan);
        }

        cudnnBackendDescriptor_t extractedEngine_ = extractedEngine->get_backend_descriptor();

        status =
            cudnn_frontend::get_attribute(m_execution_plan.engine_config->get_backend_descriptor(),
                                          /*
                                          DPCT1007:707: Migration of CUDNN_ATTR_ENGINECFG_ENGINE is not supported.
                                          */
                                          CUDNN_ATTR_ENGINECFG_ENGINE,
                                          /*
                                          DPCT1007:708: Migration of CUDNN_TYPE_BACKEND_DESCRIPTOR is not supported.
                                          */
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &elemCount,
                                          &extractedEngine_);

        if (status != 0) {
            set_error_and_throw_exception(&m_execution_plan,
                                          status,
                                          "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: GetAttribute "
                                          "CUDNN_ATTR_ENGINECFG_ENGINE Failed");
            return std::move(m_execution_plan);
        }

        m_execution_plan.buildTag(extractedEngine);
        m_execution_plan.fetchNotes(extractedEngine);
        m_execution_plan.computeWorkSpaceSize();

        getLogger() << "[cudnn_frontend] " << m_execution_plan << std::endl;
        return std::move(m_execution_plan);
    }

    explicit ExecutionPlanBuilder_v8()                       = default;
    ~ExecutionPlanBuilder_v8()                               = default;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8 &&)      = delete;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8 const &) = delete;
    ExecutionPlanBuilder_v8 &
    operator=(ExecutionPlanBuilder_v8 const &) = delete;

   private:
    ExecutionPlan_v8 m_execution_plan;
};

using ExecutionPlan        = ExecutionPlan_v8;
using ExecutionPlanBuilder = ExecutionPlanBuilder_v8;

}  // namespace cudnn_frontend
