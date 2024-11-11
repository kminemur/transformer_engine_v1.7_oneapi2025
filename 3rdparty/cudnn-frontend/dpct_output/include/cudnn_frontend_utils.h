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
#include <exception>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#ifndef CUDNN_FRONTEND_SKIP_NLOHMANN_JSON
#include "cudnn_frontend/thirdparty/nlohmann/json.hpp"
#endif

using json = nlohmann::json;

#include <iomanip>
#include <sstream>

template <>
struct nlohmann::adl_serializer<float> {
    static void
    to_json(nlohmann::json& j, const float& f) {
        // Convert float to hexadecimal string
        unsigned int intValue;
        std::memcpy(&intValue, &f, sizeof(float));

        std::stringstream stream;
        stream << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << intValue;
        j = stream.str();
    }

    static void
    from_json(const nlohmann::json& j, float& f) {
        // Read hexadecimal string and convert back to float
        std::string hexValueStr = j.get<std::string>();
        unsigned int hexValue;
        std::stringstream stream(hexValueStr);
        stream >> std::hex >> hexValue;

        std::memcpy(&f, &hexValue, sizeof(float));
    }
};

template <>
struct nlohmann::adl_serializer<sycl::half> {
    static void
    to_json(json& j, const sycl::half& opt) {
        // No precision loss when converting to float
        j = sycl::vec<sycl::half, 1>(opt).convert<float, sycl::rounding_mode::automatic>()[0];
    }

    static void
    from_json(const json& j, sycl::half& opt) {
        opt = sycl::vec<float, 1>(j.get<float>()).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
};

template <>
struct nlohmann::adl_serializer<sycl::ext::oneapi::bfloat16> {
    static void
    to_json(json& j, const sycl::ext::oneapi::bfloat16& opt) {
        // No precision loss when converting to float
        j = static_cast<float>(opt);
    }

    static void
    from_json(const json& j, sycl::ext::oneapi::bfloat16& opt) {
        opt = sycl::ext::oneapi::bfloat16(j.get<float>());
    }
};

template <>
struct nlohmann::adl_serializer<std::variant<int32_t, sycl::half, float, sycl::ext::oneapi::bfloat16>> {
    static void
    to_json(nlohmann::json& j, const std::variant<int32_t, sycl::half, float, sycl::ext::oneapi::bfloat16>& data) {
        std::visit([&](const auto& v) { j = {{"index", data.index()}, {"value", v}}; }, data);
    }

    static void
    from_json(const nlohmann::json& j, std::variant<int32_t, sycl::half, float, sycl::ext::oneapi::bfloat16>& data) {
        if (!j.is_object() || !j.contains("index") || !j.contains("value")) {
            throw std::invalid_argument("Invalid JSON format for std::variant");
        }

        size_t type_index = j.at("index").get<size_t>();
        if (type_index == 0) {
            data = j.at("value").get<int32_t>();
        } else if (type_index == 1) {
            data = j.at("value").get<sycl::half>();
        } else if (type_index == 2) {
            data = j.at("value").get<float>();
        } else if (type_index == 3) {
            data = j.at("value").get<sycl::ext::oneapi::bfloat16>();
        } else {
            throw std::out_of_range("Variant index out of range");
        }
    }
};

// Specialization of nlohmann::adl_serializer for std::optional<T>
template <typename T>
struct nlohmann::adl_serializer<std::optional<T>> {
    static void
    to_json(json& j, const std::optional<T>& opt) {
        if (opt.has_value())
            j = *opt;
        else
            j = nullptr;
    }

    static void
    from_json(const json& j, std::optional<T>& opt) {
        if (!j.is_null())
            opt = j.get<T>();
        else
            opt.reset();
    }
};

// Specialization of nlohmann::adl_serializer for std::shared_ptr<T>
template <typename T>
struct nlohmann::adl_serializer<std::shared_ptr<T>> {
    static void
    to_json(json& j, const std::shared_ptr<T>& ptr) {
        if (ptr)
            j = *ptr;
        else
            j = nullptr;
    }

    static void
    from_json(const json& j, std::shared_ptr<T>& ptr) {
        if (!j.is_null())
            ptr = std::make_shared<T>(j.get<T>());
        else
            ptr.reset();
    }
};

// Specialization of nlohmann::adl_serializer for cudnnFraction_t
template <>
struct nlohmann::adl_serializer<cudnnFraction_t> {
    static void
    to_json(json& j, const cudnnFraction_t& fraction) {
        j = fraction.numerator;
    }

    static void
    from_json(const json& j, cudnnFraction_t& fraction) {
        fraction.numerator = j;
    }
};

#include "cudnn_frontend_shim.h"
#include "cudnn_backend_base.h"
#include "cudnn_frontend_Logging.h"

#ifndef NV_CUDNN_DISABLE_EXCEPTION
#ifdef _MSC_VER
#pragma warning(disable : 4702)  // if exceptions are enabled there are unreachable return statements
#endif
#endif

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)
namespace cudnn_frontend {

/// Detailed feature_vector. Generally the Tensor and Operation properties
using feature_vector_t = std::vector<int64_t>;

class cudnnException : public std::runtime_error {
   public:
    cudnnException(const char* message, dpct::err1 status) throw() : std::runtime_error(message) {
        error_status = status;
    }
    virtual const char*
    what() const throw() {
        return std::runtime_error::what();
    }
    dpct::err1
    getCudnnStatus() {
        return error_status;
    }

    dpct::err1 error_status;
};

static inline bool
AllowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

static inline std::string
to_string(dpct::err1 const status) {
    return cudnn_frontend::get_error_string(status);
}

#ifndef NV_CUDNN_DISABLE_EXCEPTION
[[noreturn]]
#endif
static inline void
set_error_and_throw_exception(BackendDescriptor const* desc, dpct::err1 status, const char* message) {
    if (desc != nullptr) {
        desc->set_status(status);
        desc->set_error(message);
    }
#ifndef NV_CUDNN_DISABLE_EXCEPTION
    throw cudnnException(std::string(std::string(message) + std::string(" cudnn_status: ") + to_string(status)).c_str(),
                         status);
#endif
}

static inline std::string
to_string(cudnnBackendBehaviorNote_t note) {
    switch (note) {
        /*
        DPCT1007:14: Migration of CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION is not supported.
        */
        case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
            return std::string("CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION");
        /*
        DPCT1007:15: Migration of CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER is not supported.
        */
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
            return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER");
        /*
        DPCT1007:16: Migration of CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER is not supported.
        */
        case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
            return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER");
        /*
        DPCT1007:17: Migration of CUDNN_BEHAVIOR_NOTE_TYPE_COUNT is not supported.
        */
        case CUDNN_BEHAVIOR_NOTE_TYPE_COUNT:
            return std::string("CUDNN_BEHAVIOR_NOTE_TYPE_COUNT");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_BEHAVIOR_NOTE");
#endif
    }
    return std::string("INVALID_BEHAVIOR_NOTE");
}

static inline std::string
to_string(cudnnBackendNumericalNote_t note) {
    switch (note) {
        /*
        DPCT1007:18: Migration of CUDNN_NUMERICAL_NOTE_TENSOR_CORE is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
            return std::string("CUDNN_NUMERICAL_NOTE_TENSOR_CORE");
        /*
        DPCT1007:19: Migration of CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
            return std::string("CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS");
        /*
        DPCT1007:20: Migration of CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
            return std::string("CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION");
        /*
        DPCT1007:21: Migration of CUDNN_NUMERICAL_NOTE_FFT is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_FFT:
            return std::string("CUDNN_NUMERICAL_NOTE_FFT");
        /*
        DPCT1007:22: Migration of CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
            return std::string("CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC");
        /*
        DPCT1007:23: Migration of CUDNN_NUMERICAL_NOTE_WINOGRAD is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_WINOGRAD:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD");
        /*
        DPCT1007:24: Migration of CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4 is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4");
        /*
        DPCT1007:25: Migration of CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6 is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6");
        /*
        DPCT1007:26: Migration of CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13 is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
            return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13");
        /*
        DPCT1007:27: Migration of CUDNN_NUMERICAL_NOTE_TYPE_COUNT is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_TYPE_COUNT:
            return std::string("CUDNN_NUMERICAL_NOTE_TYPE_COUNT");

            // If none of the above cases hit, its definitely strict nan prop and should raise an error.
#if (CUDNN_VERSION >= 90100)
        /*
        DPCT1007:28: Migration of CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP is not supported.
        */
        case CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP:
            return std::string("CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_NUMERICAL_NOTE");
#endif
    }
    return std::string("INVALID_NUMERICAL_NOTE");
}

#if (CUDNN_VERSION >= 8700)
static inline std::string
to_string(cudnnRngDistribution_t distribution) {
    switch (distribution) {
        /*
        DPCT1007:29: Migration of CUDNN_RNG_DISTRIBUTION_BERNOULLI is not supported.
        */
        case CUDNN_RNG_DISTRIBUTION_BERNOULLI:
            return std::string("CUDNN_RNG_DISTRIBUTION_BERNOULLI");
        /*
        DPCT1007:30: Migration of CUDNN_RNG_DISTRIBUTION_UNIFORM is not supported.
        */
        case CUDNN_RNG_DISTRIBUTION_UNIFORM:
            return std::string("CUDNN_RNG_DISTRIBUTION_UNIFORM");
        /*
        DPCT1007:31: Migration of CUDNN_RNG_DISTRIBUTION_NORMAL is not supported.
        */
        case CUDNN_RNG_DISTRIBUTION_NORMAL:
            return std::string("CUDNN_RNG_DISTRIBUTION_NORMAL");
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return std::string("UNKNOWN_CUDNN_DISTRIBUTION");
#endif
    }
    return std::string("");
}
#endif

enum class BuildPlanPolicy_t {
    // Builds and stores the "first successful" plan from the list returned by heuristics.
    // heuristics list is traversed sequentially and in decreasing order of potential performance.
    HEURISTICS_CHOICE,
    // Builds and stores all the "successful" plans from the list returned by heuristics.
    ALL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(BuildPlanPolicy_t,
                             {
                                 {BuildPlanPolicy_t::HEURISTICS_CHOICE, "HEURISTICS_CHOICE"},
                                 {BuildPlanPolicy_t::ALL, "ALL"},
                             })

enum class TensorReordering_t {
    NONE,
    INT8x32,
    F16x16,
};

NLOHMANN_JSON_SERIALIZE_ENUM(TensorReordering_t,
                             {
                                 {TensorReordering_t::NONE, "NONE"},
                                 {TensorReordering_t::INT8x32, "INT8x32"},
                                 {TensorReordering_t::F16x16, "F16x16"},
                             })

enum class ResampleMode_t {
    NOT_SET,

    AVGPOOL_EXCLUDE_PADDING,
    AVGPOOL_INCLUDE_PADDING,
    BILINEAR,
    NEAREST,
    MAXPOOL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ResampleMode_t,
                             {
                                 {ResampleMode_t::NOT_SET, nullptr},
                                 {ResampleMode_t::AVGPOOL_EXCLUDE_PADDING, "AVGPOOL_EXCLUDE_PADDING"},
                                 {ResampleMode_t::AVGPOOL_INCLUDE_PADDING, "AVGPOOL_INCLUDE_PADDING"},
                                 {ResampleMode_t::BILINEAR, "BILINEAR"},
                                 {ResampleMode_t::NEAREST, "NEAREST"},
                                 {ResampleMode_t::MAXPOOL, "MAXPOOL"},
                             })

enum class PaddingMode_t {
    NOT_SET,

    EDGE_VAL_PAD,
    NEG_INF_PAD,
    ZERO_PAD
};

NLOHMANN_JSON_SERIALIZE_ENUM(PaddingMode_t,
                             {
                                 {PaddingMode_t::NOT_SET, nullptr},
                                 {PaddingMode_t::EDGE_VAL_PAD, "EDGE_VAL_PAD"},
                                 {PaddingMode_t::NEG_INF_PAD, "NEG_INF_PAD"},
                                 {PaddingMode_t::ZERO_PAD, "ZERO_PAD"},
                             })

enum class NormFwdPhase_t {
    NOT_SET,

    INFERENCE,
    TRAINING
};

NLOHMANN_JSON_SERIALIZE_ENUM(NormFwdPhase_t,
                             {
                                 {NormFwdPhase_t::NOT_SET, nullptr},
                                 {NormFwdPhase_t::INFERENCE, "INFERENCE"},
                                 {NormFwdPhase_t::TRAINING, "TRAINING"},
                             })

enum class DescriptorType_t {
    NOT_SET,

    POINTWISE_DESCRIPTOR,
    CONVOLUTION_DESCRIPTOR,
    ENGINE_DESCRIPTOR,
    ENGINECFG_DESCRIPTOR,
    ENGINEHEUR_DESCRIPTOR,
    EXECUTION_PLAN_DESCRIPTOR,
    INTERMEDIATE_INFO_DESCRIPTOR,
    KNOB_CHOICE_DESCRIPTOR,
    KNOB_INFO_DESCRIPTOR,
    LAYOUT_INFO_DESCRIPTOR,
    OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
    OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
    OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
    OPERATION_POINTWISE_DESCRIPTOR,
    OPERATION_GEN_STATS_DESCRIPTOR,
    OPERATIONGRAPH_DESCRIPTOR,
    VARIANT_PACK_DESCRIPTOR,
    TENSOR_DESCRIPTOR,
    MATMUL_DESCRIPTOR,
    OPERATION_MATMUL_DESCRIPTOR,
    OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
    REDUCTION_DESCRIPTOR,
    OPERATION_REDUCTION_DESCRIPTOR,
    OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
    RESAMPLE_DESCRIPTOR,
    OPERATION_RESAMPLE_FWD_DESCRIPTOR,
    OPERATION_RESAMPLE_BWD_DESCRIPTOR,
    OPERATION_CONCAT_DESCRIPTOR,
    OPERATION_SIGNAL_DESCRIPTOR,
    OPERATION_NORM_FORWARD_DESCRIPTOR,
    OPERATION_NORM_BACKWARD_DESCRIPTOR,
    OPERATION_RESHAPE_DESCRIPTOR,
    RNG_DESCRIPTOR,
    OPERATION_RNG_DESCRIPTOR
};

enum class NormMode_t {
    NOT_SET,

    LAYER_NORM,
    INSTANCE_NORM,
    BATCH_NORM,
    GROUP_NORM,
    RMS_NORM,
};

NLOHMANN_JSON_SERIALIZE_ENUM(NormMode_t,
                             {
                                 {NormMode_t::NOT_SET, nullptr},
                                 {NormMode_t::LAYER_NORM, "LAYER_NORM"},
                                 {NormMode_t::INSTANCE_NORM, "INSTANCE_NORM"},
                                 {NormMode_t::BATCH_NORM, "BATCH_NORM"},
                                 {NormMode_t::GROUP_NORM, "GROUP_NORM"},
                                 {NormMode_t::RMS_NORM, "RMS_NORM"},
                             })

enum class PointwiseMode_t {
    NOT_SET,

    ADD,
    MUL,
    SQRT,
    MAX,
    MIN,
    RELU_FWD,
    TANH_FWD,
    SIGMOID_FWD,
    ELU_FWD,
    GELU_FWD,
    SOFTPLUS_FWD,
    SWISH_FWD,
    RELU_BWD,
    TANH_BWD,
    SIGMOID_BWD,
    ELU_BWD,
    GELU_BWD,
    SOFTPLUS_BWD,
    SWISH_BWD,
    ERF,
    IDENTITY,
    GELU_APPROX_TANH_BWD,
    GELU_APPROX_TANH_FWD,
    GEN_INDEX,
    BINARY_SELECT,
    EXP,
    LOG,
    NEG,
    MOD,
    POW,
    ABS,
    CEIL,
    COS,
    FLOOR,
    RSQRT,
    SIN,
    LOGICAL_NOT,
    TAN,
    SUB,
    ADD_SQUARE,
    DIV,
    CMP_EQ,
    CMP_NEQ,
    CMP_GT,
    CMP_GE,
    CMP_LT,
    CMP_LE,
    LOGICAL_AND,
    LOGICAL_OR,
    RECIPROCAL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(PointwiseMode_t,
                             {
                                 {PointwiseMode_t::NOT_SET, nullptr},
                                 {PointwiseMode_t::ADD, "ADD"},
                                 {PointwiseMode_t::MUL, "MUL"},
                                 {PointwiseMode_t::SQRT, "SQRT"},
                                 {PointwiseMode_t::MAX, "MAX"},
                                 {PointwiseMode_t::MIN, "MIN"},
                                 {PointwiseMode_t::RELU_FWD, "RELU_FWD"},
                                 {PointwiseMode_t::TANH_FWD, "TANH_FWD"},
                                 {PointwiseMode_t::SIGMOID_FWD, "SIGMOID_FWD"},
                                 {PointwiseMode_t::ELU_FWD, "ELU_FWD"},
                                 {PointwiseMode_t::GELU_FWD, "GELU_FWD"},
                                 {PointwiseMode_t::SOFTPLUS_FWD, "SOFTPLUS_FWD"},
                                 {PointwiseMode_t::SWISH_FWD, "SWISH_FWD"},
                                 {PointwiseMode_t::RELU_BWD, "RELU_BWD"},
                                 {PointwiseMode_t::TANH_BWD, "TANH_BWD"},
                                 {PointwiseMode_t::SIGMOID_BWD, "SIGMOID_BWD"},
                                 {PointwiseMode_t::ELU_BWD, "ELU_BWD"},
                                 {PointwiseMode_t::GELU_BWD, "GELU_BWD"},
                                 {PointwiseMode_t::SOFTPLUS_BWD, "SOFTPLUS_BWD"},
                                 {PointwiseMode_t::SWISH_BWD, "SWISH_BWD"},
                                 {PointwiseMode_t::ERF, "ERF"},
                                 {PointwiseMode_t::IDENTITY, "IDENTITY"},
                                 {PointwiseMode_t::GELU_APPROX_TANH_BWD, "GELU_APPROX_TANH_BWD"},
                                 {PointwiseMode_t::GELU_APPROX_TANH_FWD, "GELU_APPROX_TANH_FWD"},
                                 {PointwiseMode_t::GEN_INDEX, "GEN_INDEX"},
                                 {PointwiseMode_t::BINARY_SELECT, "BINARY_SELECT"},
                                 {PointwiseMode_t::EXP, "EXP"},
                                 {PointwiseMode_t::LOG, "LOG"},
                                 {PointwiseMode_t::NEG, "NEG"},
                                 {PointwiseMode_t::MOD, "MOD"},
                                 {PointwiseMode_t::POW, "POW"},
                                 {PointwiseMode_t::ABS, "ABS"},
                                 {PointwiseMode_t::CEIL, "CEIL"},
                                 {PointwiseMode_t::COS, "COS"},
                                 {PointwiseMode_t::FLOOR, "FLOOR"},
                                 {PointwiseMode_t::RSQRT, "RSQRT"},
                                 {PointwiseMode_t::SIN, "SIN"},
                                 {PointwiseMode_t::LOGICAL_NOT, "LOGICAL_NOT"},
                                 {PointwiseMode_t::TAN, "TAN"},
                                 {PointwiseMode_t::SUB, "SUB"},
                                 {PointwiseMode_t::ADD_SQUARE, "ADD_SQUARE"},
                                 {PointwiseMode_t::DIV, "DIV"},
                                 {PointwiseMode_t::CMP_EQ, "CMP_EQ"},
                                 {PointwiseMode_t::CMP_NEQ, "CMP_NEQ"},
                                 {PointwiseMode_t::CMP_GT, "CMP_GT"},
                                 {PointwiseMode_t::CMP_GE, "CMP_GE"},
                                 {PointwiseMode_t::CMP_LT, "CMP_LT"},
                                 {PointwiseMode_t::CMP_LE, "CMP_LE"},
                                 {PointwiseMode_t::LOGICAL_AND, "LOGICAL_AND"},
                                 {PointwiseMode_t::LOGICAL_OR, "LOGICAL_OR"},
                                 {PointwiseMode_t::RECIPROCAL, "RECIPROCAL"},
                             })

enum class HeurMode_t {
    A,
    B,
    FALLBACK,
};

NLOHMANN_JSON_SERIALIZE_ENUM(HeurMode_t,
                             {
                                 {HeurMode_t::A, "A"},
                                 {HeurMode_t::B, "B"},
                                 {HeurMode_t::FALLBACK, "FALLBACK"},
                             })

enum class BehaviorNote_t {
    RUNTIME_COMPILATION,
    REQUIRES_FILTER_INT8x32_REORDER,
    REQUIRES_BIAS_INT8x32_REORDER,
};

NLOHMANN_JSON_SERIALIZE_ENUM(BehaviorNote_t,
                             {
                                 {BehaviorNote_t::RUNTIME_COMPILATION, "RUNTIME_COMPILATION"},
                                 {BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER, "REQUIRES_FILTER_INT8x32_REORDER"},
                                 {BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER, "REQUIRES_BIAS_INT8x32_REORDER"},
                             })

enum class NumericalNote_t {
    TENSOR_CORE,
    DOWN_CONVERT_INPUTS,
    REDUCED_PRECISION_REDUCTION,
    FFT,
    NONDETERMINISTIC,
    WINOGRAD,
    WINOGRAD_TILE_4x4,
    WINOGRAD_TILE_6x6,
    WINOGRAD_TILE_13x13,
    STRICT_NAN_PROP,
};

NLOHMANN_JSON_SERIALIZE_ENUM(NumericalNote_t,
                             {
                                 {NumericalNote_t::TENSOR_CORE, "TENSOR_CORE"},
                                 {NumericalNote_t::DOWN_CONVERT_INPUTS, "DOWN_CONVERT_INPUTS"},
                                 {NumericalNote_t::REDUCED_PRECISION_REDUCTION, "REDUCED_PRECISION_REDUCTION"},
                                 {NumericalNote_t::FFT, "FFT"},
                                 {NumericalNote_t::NONDETERMINISTIC, "NONDETERMINISTIC"},
                                 {NumericalNote_t::WINOGRAD, "WINOGRAD"},
                                 {NumericalNote_t::WINOGRAD_TILE_4x4, "WINOGRAD_TILE_4x4"},
                                 {NumericalNote_t::WINOGRAD_TILE_6x6, "WINOGRAD_TILE_6x6"},
                                 {NumericalNote_t::WINOGRAD_TILE_13x13, "WINOGRAD_TILE_13x13"},
                                 {NumericalNote_t::STRICT_NAN_PROP, "STRICT_NAN_PROP"},
                             })

enum class DataType_t {
    NOT_SET,

    FLOAT,
    DOUBLE,
    HALF,
    INT8,
    INT32,
    INT8x4,
    UINT8,
    UINT8x4,
    INT8x32,
    BFLOAT16,
    INT64,
    BOOLEAN,
    FP8_E4M3,
    FP8_E5M2,
    FAST_FLOAT_FOR_FP8,
};

NLOHMANN_JSON_SERIALIZE_ENUM(DataType_t,
                             {
                                 {DataType_t::NOT_SET, nullptr},
                                 {DataType_t::FLOAT, "FLOAT"},
                                 {DataType_t::DOUBLE, "DOUBLE"},
                                 {DataType_t::HALF, "HALF"},
                                 {DataType_t::INT8, "INT8"},
                                 {DataType_t::INT32, "INT32"},
                                 {DataType_t::INT8x4, "INT8x4"},
                                 {DataType_t::UINT8, "UINT8"},
                                 {DataType_t::UINT8x4, "UINT8x4"},
                                 {DataType_t::INT8x32, "INT8x32"},
                                 {DataType_t::BFLOAT16, "BFLOAT16"},
                                 {DataType_t::INT64, "INT64"},
                                 {DataType_t::BOOLEAN, "BOOLEAN"},
                                 {DataType_t::FP8_E4M3, "FP8_E4M3"},
                                 {DataType_t::FP8_E5M2, "FP8_E5M2"},
                                 {DataType_t::FAST_FLOAT_FOR_FP8, "FAST_FLOAT_FOR_FP8"},
                             })

enum class ReductionMode_t {
    NOT_SET,

    ADD,
    MUL,
    MIN,
    MAX,
    AMAX,
    AVG,
    NORM1,
    NORM2,
    MUL_NO_ZEROS
};

NLOHMANN_JSON_SERIALIZE_ENUM(ReductionMode_t,
                             {
                                 {ReductionMode_t::NOT_SET, nullptr},
                                 {ReductionMode_t::ADD, "ADD"},
                                 {ReductionMode_t::MUL, "MUL"},
                                 {ReductionMode_t::MIN, "MIN"},
                                 {ReductionMode_t::MAX, "MAX"},
                                 {ReductionMode_t::AMAX, "AMAX"},
                                 {ReductionMode_t::AVG, "AVG"},
                                 {ReductionMode_t::NORM1, "NORM1"},
                                 {ReductionMode_t::NORM2, "NORM2"},
                                 {ReductionMode_t::MUL_NO_ZEROS, "MUL_NO_ZEROS"},
                             })

enum class RngDistribution_t {
    NOT_SET,

    BERNOULLI,
    UNIFORM,
    NORMAL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(RngDistribution_t,
                             {
                                 {RngDistribution_t::NOT_SET, nullptr},
                                 {RngDistribution_t::BERNOULLI, "BERNOULLI"},
                                 {RngDistribution_t::UNIFORM, "UNIFORM"},
                                 {RngDistribution_t::NORMAL, "NORMAL"},
                             })

static int64_t
get_pointwise_mode_port_count(PointwiseMode_t const& mode) {
    switch (mode) {
        case PointwiseMode_t::NOT_SET:
            return 0;

        case PointwiseMode_t::ADD:
        case PointwiseMode_t::MUL:
        case PointwiseMode_t::DIV:
        case PointwiseMode_t::ADD_SQUARE:
        case PointwiseMode_t::SUB:
        case PointwiseMode_t::CMP_EQ:
        case PointwiseMode_t::CMP_NEQ:
        case PointwiseMode_t::CMP_GT:
        case PointwiseMode_t::CMP_GE:
        case PointwiseMode_t::CMP_LT:
        case PointwiseMode_t::CMP_LE:
        case PointwiseMode_t::LOGICAL_AND:
        case PointwiseMode_t::LOGICAL_OR:
        case PointwiseMode_t::MIN:
        case PointwiseMode_t::MAX:
        case PointwiseMode_t::MOD:
        case PointwiseMode_t::RELU_BWD:
        case PointwiseMode_t::TANH_BWD:
        case PointwiseMode_t::SIGMOID_BWD:
        case PointwiseMode_t::ELU_BWD:
        case PointwiseMode_t::GELU_BWD:
        case PointwiseMode_t::SOFTPLUS_BWD:
        case PointwiseMode_t::SWISH_BWD:
        case PointwiseMode_t::GELU_APPROX_TANH_BWD:
        case PointwiseMode_t::POW:
            return 3;

        case PointwiseMode_t::SQRT:
        case PointwiseMode_t::RELU_FWD:
        case PointwiseMode_t::TANH_FWD:
        case PointwiseMode_t::SIGMOID_FWD:
        case PointwiseMode_t::ELU_FWD:
        case PointwiseMode_t::GELU_FWD:
        case PointwiseMode_t::SOFTPLUS_FWD:
        case PointwiseMode_t::SWISH_FWD:
        case PointwiseMode_t::EXP:
        case PointwiseMode_t::LOG:
        case PointwiseMode_t::NEG:
        case PointwiseMode_t::ABS:
        case PointwiseMode_t::CEIL:
        case PointwiseMode_t::FLOOR:
        case PointwiseMode_t::COS:
        case PointwiseMode_t::TAN:
        case PointwiseMode_t::SIN:
        case PointwiseMode_t::RSQRT:
        case PointwiseMode_t::LOGICAL_NOT:
        case PointwiseMode_t::GEN_INDEX:
        case PointwiseMode_t::ERF:
        case PointwiseMode_t::GELU_APPROX_TANH_FWD:
        case PointwiseMode_t::IDENTITY:
        case PointwiseMode_t::RECIPROCAL:
            return 2;

        case PointwiseMode_t::BINARY_SELECT:
            return 4;
    }
    return -1;
}

static inline std::ostream&
operator<<(std::ostream& os, const DescriptorType_t& mode) {
    switch (mode) {
        case DescriptorType_t::POINTWISE_DESCRIPTOR:
            os << "POINTWISE_DESCRIPTOR";
            break;
        case DescriptorType_t::CONVOLUTION_DESCRIPTOR:
            os << "CONVOLUTION_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINE_DESCRIPTOR:
            os << "ENGINE_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINECFG_DESCRIPTOR:
            os << "ENGINECFG_DESCRIPTOR";
            break;
        case DescriptorType_t::ENGINEHEUR_DESCRIPTOR:
            os << "ENGINEHEUR_DESCRIPTOR";
            break;
        case DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR:
            os << "EXECUTION_PLAN_DESCRIPTOR";
            break;
        case DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR:
            os << "INTERMEDIATE_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::KNOB_CHOICE_DESCRIPTOR:
            os << "KNOB_CHOICE_DESCRIPTOR";
            break;
        case DescriptorType_t::KNOB_INFO_DESCRIPTOR:
            os << "KNOB_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::LAYOUT_INFO_DESCRIPTOR:
            os << "LAYOUT_INFO_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            os << "OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR:
            os << "OPERATION_POINTWISE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR:
            os << "OPERATION_GEN_STATS_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR:
            os << "OPERATIONGRAPH_DESCRIPTOR";
            break;
        case DescriptorType_t::VARIANT_PACK_DESCRIPTOR:
            os << "VARIANT_PACK_DESCRIPTOR";
            break;
        case DescriptorType_t::TENSOR_DESCRIPTOR:
            os << "TENSOR_DESCRIPTOR";
            break;
        case DescriptorType_t::MATMUL_DESCRIPTOR:
            os << "MATMUL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR:
            os << "OPERATION_MATMUL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
            os << "OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR";
            break;
        case DescriptorType_t::REDUCTION_DESCRIPTOR:
            os << "REDUCTION_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR:
            os << "OPERATION_REDUCTION_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            os << "OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR";
            break;
        case DescriptorType_t::RESAMPLE_DESCRIPTOR:
            os << "RESAMPLE_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            os << "OPERATION_RESAMPLE_FWD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR:
            os << "OPERATION_RESAMPLE_BWD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR:
            os << "OPERATION_CONCAT_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR:
            os << "OPERATION_SIGNAL_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR:
            os << "OPERATION_NORM_FORWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR:
            os << "OPERATION_NORM_BACKWARD_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR:
            os << "OPERATION_RESHAPE_DESCRIPTOR";
            break;
        case DescriptorType_t::RNG_DESCRIPTOR:
            os << "RNG_DESCRIPTOR";
            break;
        case DescriptorType_t::OPERATION_RNG_DESCRIPTOR:
            os << "OPERATION_RNG_DESCRIPTOR";
            break;
        case DescriptorType_t::NOT_SET:
            os << "NOT_SET";
            break;
    }
    return os;
}

namespace detail {

inline std::vector<float>
get_abili_slope(int64_t const n_heads) {
    std::vector<float> slope;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)  // this could be ommited with c++17 and contexpr
#endif
    int n = 1 << static_cast<int>(log2(static_cast<double>(n_heads)));
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    for (int i = 0; i < n; i++) {
        slope.push_back((float)(i + 1.0));
    }

    for (int i = 0; i < 2 * (n_heads - n); i += 2) {
        slope.push_back(static_cast<float>(i + 1) * 0.5f);
    }

    for (float& elem : slope) {
        elem *= -8.0f;
        elem /= static_cast<float>(n);
        elem = powf(2.0, elem);
    }

    return slope;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::DataType_t const mode, dpct::library_data_t& cudnn_mode) {
    switch (mode) {
        case DataType_t::FLOAT:
            cudnn_mode = dpct::library_data_t::real_float;
            return 0;
        case DataType_t::DOUBLE:
            cudnn_mode = dpct::library_data_t::real_double;
            return 0;
        case DataType_t::HALF:
            cudnn_mode = dpct::library_data_t::real_half;
            return 0;
        case DataType_t::INT8:
            cudnn_mode = dpct::library_data_t::real_int8;
            return 0;
        case DataType_t::INT32:
            cudnn_mode = dpct::library_data_t::real_int32;
            return 0;
        case DataType_t::INT8x4:
            cudnn_mode = dpct::library_data_t::real_int8_4;
            return 0;
        case DataType_t::UINT8:
            cudnn_mode = dpct::library_data_t::real_uint8;
            return 0;
        case DataType_t::UINT8x4:
            cudnn_mode = dpct::library_data_t::real_uint8_4;
            return 0;
        case DataType_t::INT8x32:
            cudnn_mode = dpct::library_data_t::real_int8_32;
            return 0;
        case DataType_t::BFLOAT16:
            cudnn_mode = dpct::library_data_t::real_bfloat16;
            return 0;
        case DataType_t::INT64:
            /*
            DPCT1007:32: Migration of CUDNN_DATA_INT64 is not supported.
            */
            cudnn_mode = CUDNN_DATA_INT64;
            return 0;
        case DataType_t::BOOLEAN:
            /*
            DPCT1007:33: Migration of CUDNN_DATA_BOOLEAN is not supported.
            */
            cudnn_mode = CUDNN_DATA_BOOLEAN;
            return 0;
        case DataType_t::FP8_E4M3:
#if (CUDNN_VERSION >= 8600)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8600, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:34: Migration of CUDNN_DATA_FP8_E4M3 is not supported.
            */
            cudnn_mode = CUDNN_DATA_FP8_E4M3;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::FP8_E5M2:
#if (CUDNN_VERSION >= 8600)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8600, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:35: Migration of CUDNN_DATA_FP8_E5M2 is not supported.
            */
            cudnn_mode = CUDNN_DATA_FP8_E5M2;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DataType_t::FAST_FLOAT_FOR_FP8:
#if (CUDNN_VERSION >= 8700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:36: Migration of CUDNN_DATA_FAST_FLOAT_FOR_FP8 is not supported.
            */
            cudnn_mode = CUDNN_DATA_FAST_FLOAT_FOR_FP8;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::ReductionMode_t const mode, dpct::dnnl::reduction_op& cudnn_mode) {
    switch (mode) {
        case ReductionMode_t::ADD:
            cudnn_mode = dpct::dnnl::reduction_op::sum;
            return 0;
        case ReductionMode_t::MUL:
            cudnn_mode = dpct::dnnl::reduction_op::mul;
            return 0;
        case ReductionMode_t::MIN:
            cudnn_mode = dpct::dnnl::reduction_op::min;
            return 0;
        case ReductionMode_t::MAX:
            cudnn_mode = dpct::dnnl::reduction_op::max;
            return 0;
        case ReductionMode_t::AMAX:
            cudnn_mode = dpct::dnnl::reduction_op::amax;
            return 0;
        case ReductionMode_t::AVG:
            cudnn_mode = dpct::dnnl::reduction_op::mean;
            return 0;
        case ReductionMode_t::NORM1:
            cudnn_mode = dpct::dnnl::reduction_op::norm1;
            return 0;
        case ReductionMode_t::NORM2:
            cudnn_mode = dpct::dnnl::reduction_op::norm2;
            return 0;
        case ReductionMode_t::MUL_NO_ZEROS:
            cudnn_mode = dpct::dnnl::reduction_op::mul_no_zeros;
            return 0;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::PointwiseMode_t const mode, cudnnPointwiseMode_t& cudnn_mode) {
    switch (mode) {
        case PointwiseMode_t::ADD:
            /*
            DPCT1007:37: Migration of CUDNN_POINTWISE_ADD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_ADD;
            return 0;
        case PointwiseMode_t::MUL:
            /*
            DPCT1007:38: Migration of CUDNN_POINTWISE_MUL is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_MUL;
            return 0;
        case PointwiseMode_t::SQRT:
            /*
            DPCT1007:39: Migration of CUDNN_POINTWISE_SQRT is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SQRT;
            return 0;
        case PointwiseMode_t::MAX:
            /*
            DPCT1007:40: Migration of CUDNN_POINTWISE_MAX is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_MAX;
            return 0;
        case PointwiseMode_t::MIN:
            /*
            DPCT1007:41: Migration of CUDNN_POINTWISE_MIN is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_MIN;
            return 0;
        case PointwiseMode_t::RELU_FWD:
            /*
            DPCT1007:42: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_RELU_FWD;
            return 0;
        case PointwiseMode_t::TANH_FWD:
            /*
            DPCT1007:43: Migration of CUDNN_POINTWISE_TANH_FWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_TANH_FWD;
            return 0;
        case PointwiseMode_t::SIGMOID_FWD:
            /*
            DPCT1007:44: Migration of CUDNN_POINTWISE_SIGMOID_FWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SIGMOID_FWD;
            return 0;
        case PointwiseMode_t::ELU_FWD:
            /*
            DPCT1007:45: Migration of CUDNN_POINTWISE_ELU_FWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_ELU_FWD;
            return 0;
        case PointwiseMode_t::GELU_FWD:
            /*
            DPCT1007:46: Migration of CUDNN_POINTWISE_GELU_FWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_GELU_FWD;
            return 0;
        case PointwiseMode_t::SOFTPLUS_FWD:
            /*
            DPCT1007:47: Migration of CUDNN_POINTWISE_SOFTPLUS_FWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SOFTPLUS_FWD;
            return 0;
        case PointwiseMode_t::SWISH_FWD:
            /*
            DPCT1007:48: Migration of CUDNN_POINTWISE_SWISH_FWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SWISH_FWD;
            return 0;
        case PointwiseMode_t::RELU_BWD:
            /*
            DPCT1007:49: Migration of CUDNN_POINTWISE_RELU_BWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_RELU_BWD;
            return 0;
        case PointwiseMode_t::TANH_BWD:
            /*
            DPCT1007:50: Migration of CUDNN_POINTWISE_TANH_BWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_TANH_BWD;
            return 0;
        case PointwiseMode_t::SIGMOID_BWD:
            /*
            DPCT1007:51: Migration of CUDNN_POINTWISE_SIGMOID_BWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SIGMOID_BWD;
            return 0;
        case PointwiseMode_t::ELU_BWD:
            /*
            DPCT1007:52: Migration of CUDNN_POINTWISE_ELU_BWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_ELU_BWD;
            return 0;
        case PointwiseMode_t::GELU_BWD:
            /*
            DPCT1007:53: Migration of CUDNN_POINTWISE_GELU_BWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_GELU_BWD;
            return 0;
        case PointwiseMode_t::SOFTPLUS_BWD:
            /*
            DPCT1007:54: Migration of CUDNN_POINTWISE_SOFTPLUS_BWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SOFTPLUS_BWD;
            return 0;
        case PointwiseMode_t::SWISH_BWD:
            /*
            DPCT1007:55: Migration of CUDNN_POINTWISE_SWISH_BWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SWISH_BWD;
            return 0;
        case PointwiseMode_t::DIV:
            /*
            DPCT1007:56: Migration of CUDNN_POINTWISE_DIV is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_DIV;
            return 0;
        case PointwiseMode_t::ADD_SQUARE:
            /*
            DPCT1007:57: Migration of CUDNN_POINTWISE_ADD_SQUARE is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_ADD_SQUARE;
            return 0;
        case PointwiseMode_t::EXP:
            /*
            DPCT1007:58: Migration of CUDNN_POINTWISE_EXP is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_EXP;
            return 0;
        case PointwiseMode_t::SUB:
            /*
            DPCT1007:59: Migration of CUDNN_POINTWISE_SUB is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SUB;
            return 0;
        case PointwiseMode_t::CMP_EQ:
            /*
            DPCT1007:60: Migration of CUDNN_POINTWISE_CMP_EQ is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_CMP_EQ;
            return 0;
        case PointwiseMode_t::CMP_NEQ:
            /*
            DPCT1007:61: Migration of CUDNN_POINTWISE_CMP_NEQ is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_CMP_NEQ;
            return 0;
        case PointwiseMode_t::CMP_GT:
            /*
            DPCT1007:62: Migration of CUDNN_POINTWISE_CMP_GT is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_CMP_GT;
            return 0;
        case PointwiseMode_t::CMP_GE:
            /*
            DPCT1007:63: Migration of CUDNN_POINTWISE_CMP_GE is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_CMP_GE;
            return 0;
        case PointwiseMode_t::CMP_LT:
            /*
            DPCT1007:64: Migration of CUDNN_POINTWISE_CMP_LT is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_CMP_LT;
            return 0;
        case PointwiseMode_t::CMP_LE:
            /*
            DPCT1007:65: Migration of CUDNN_POINTWISE_CMP_LE is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_CMP_LE;
            return 0;
        case PointwiseMode_t::LOGICAL_AND:
            /*
            DPCT1007:66: Migration of CUDNN_POINTWISE_LOGICAL_AND is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_LOGICAL_AND;
            return 0;
        case PointwiseMode_t::LOGICAL_OR:
            /*
            DPCT1007:67: Migration of CUDNN_POINTWISE_LOGICAL_OR is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_LOGICAL_OR;
            return 0;
        case PointwiseMode_t::LOGICAL_NOT:
            /*
            DPCT1007:68: Migration of CUDNN_POINTWISE_LOGICAL_NOT is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_LOGICAL_NOT;
            return 0;
        case PointwiseMode_t::LOG:
            /*
            DPCT1007:69: Migration of CUDNN_POINTWISE_LOG is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_LOG;
            return 0;
        case PointwiseMode_t::NEG:
            /*
            DPCT1007:70: Migration of CUDNN_POINTWISE_NEG is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_NEG;
            return 0;
        case PointwiseMode_t::MOD:
            /*
            DPCT1007:71: Migration of CUDNN_POINTWISE_MOD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_MOD;
            return 0;
        case PointwiseMode_t::POW:
            /*
            DPCT1007:72: Migration of CUDNN_POINTWISE_POW is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_POW;
            return 0;
        case PointwiseMode_t::ABS:
            /*
            DPCT1007:73: Migration of CUDNN_POINTWISE_ABS is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_ABS;
            return 0;
        case PointwiseMode_t::CEIL:
            /*
            DPCT1007:74: Migration of CUDNN_POINTWISE_CEIL is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_CEIL;
            return 0;
        case PointwiseMode_t::COS:
            /*
            DPCT1007:75: Migration of CUDNN_POINTWISE_COS is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_COS;
            return 0;
        case PointwiseMode_t::FLOOR:
            /*
            DPCT1007:76: Migration of CUDNN_POINTWISE_FLOOR is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_FLOOR;
            return 0;
        case PointwiseMode_t::RSQRT:
            /*
            DPCT1007:77: Migration of CUDNN_POINTWISE_RSQRT is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_RSQRT;
            return 0;
        case PointwiseMode_t::SIN:
            /*
            DPCT1007:78: Migration of CUDNN_POINTWISE_SIN is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_SIN;
            return 0;
        case PointwiseMode_t::TAN:
            /*
            DPCT1007:79: Migration of CUDNN_POINTWISE_TAN is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_TAN;
            return 0;
        case PointwiseMode_t::GEN_INDEX:
            /*
            DPCT1007:80: Migration of CUDNN_POINTWISE_GEN_INDEX is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_GEN_INDEX;
            return 0;
        case PointwiseMode_t::BINARY_SELECT:
            /*
            DPCT1007:81: Migration of CUDNN_POINTWISE_BINARY_SELECT is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_BINARY_SELECT;
            return 0;
        case PointwiseMode_t::ERF:
            /*
            DPCT1007:82: Migration of CUDNN_POINTWISE_ERF is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_ERF;
            return 0;
        case PointwiseMode_t::IDENTITY:
            /*
            DPCT1007:83: Migration of CUDNN_POINTWISE_IDENTITY is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_IDENTITY;
            return 0;
        case PointwiseMode_t::GELU_APPROX_TANH_BWD:
            /*
            DPCT1007:84: Migration of CUDNN_POINTWISE_GELU_APPROX_TANH_BWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_GELU_APPROX_TANH_BWD;
            return 0;
        case PointwiseMode_t::GELU_APPROX_TANH_FWD:
            /*
            DPCT1007:85: Migration of CUDNN_POINTWISE_GELU_APPROX_TANH_FWD is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_GELU_APPROX_TANH_FWD;
            return 0;
        case PointwiseMode_t::RECIPROCAL:
#if (CUDNN_VERSION >= 8900)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8900, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:86: Migration of CUDNN_POINTWISE_RECIPROCAL is not supported.
            */
            cudnn_mode = CUDNN_POINTWISE_RECIPROCAL;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::NumericalNote_t const mode, cudnnBackendNumericalNote_t& cudnn_mode) {
    switch (mode) {
        case NumericalNote_t::TENSOR_CORE:
            /*
            DPCT1007:87: Migration of CUDNN_NUMERICAL_NOTE_TENSOR_CORE is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_TENSOR_CORE;
            return 0;
        case NumericalNote_t::DOWN_CONVERT_INPUTS:
            /*
            DPCT1007:88: Migration of CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS;
            return 0;
        case NumericalNote_t::REDUCED_PRECISION_REDUCTION:
            /*
            DPCT1007:89: Migration of CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION;
            return 0;
        case NumericalNote_t::FFT:
            /*
            DPCT1007:90: Migration of CUDNN_NUMERICAL_NOTE_FFT is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_FFT;
            return 0;
        case NumericalNote_t::NONDETERMINISTIC:
            /*
            DPCT1007:91: Migration of CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC;
            return 0;
        case NumericalNote_t::WINOGRAD:
            /*
            DPCT1007:92: Migration of CUDNN_NUMERICAL_NOTE_WINOGRAD is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_WINOGRAD;
            return 0;
        case NumericalNote_t::WINOGRAD_TILE_4x4:
            /*
            DPCT1007:93: Migration of CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4 is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4;
            return 0;
        case NumericalNote_t::WINOGRAD_TILE_6x6:
            /*
            DPCT1007:94: Migration of CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6 is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6;
            return 0;
        case NumericalNote_t::WINOGRAD_TILE_13x13:
            /*
            DPCT1007:95: Migration of CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13 is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13;
            return 0;
        case NumericalNote_t::STRICT_NAN_PROP:
#if (CUDNN_VERSION >= 90100)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(90100, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:96: Migration of CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP is not supported.
            */
            cudnn_mode = CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::BehaviorNote_t const mode, cudnnBackendBehaviorNote_t& cudnn_mode) {
    switch (mode) {
        case BehaviorNote_t::RUNTIME_COMPILATION:
            /*
            DPCT1007:97: Migration of CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION is not supported.
            */
            cudnn_mode = CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
            return 0;
        case BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER:
            /*
            DPCT1007:98: Migration of CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER is not supported.
            */
            cudnn_mode = CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER;
            return 0;
        case BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER:
            /*
            DPCT1007:99: Migration of CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER is not supported.
            */
            cudnn_mode = CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER;
            return 0;
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::DescriptorType_t const mode, cudnnBackendDescriptorType_t& cudnn_mode) {
    switch (mode) {
        case DescriptorType_t::POINTWISE_DESCRIPTOR:
            /*
            DPCT1007:100: Migration of CUDNN_BACKEND_POINTWISE_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_POINTWISE_DESCRIPTOR;
            return 0;
        case DescriptorType_t::CONVOLUTION_DESCRIPTOR:
            /*
            DPCT1007:101: Migration of CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR;
            return 0;
        case DescriptorType_t::ENGINE_DESCRIPTOR:
            /*
            DPCT1007:102: Migration of CUDNN_BACKEND_ENGINE_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
            return 0;
        case DescriptorType_t::ENGINECFG_DESCRIPTOR:
            /*
            DPCT1007:103: Migration of CUDNN_BACKEND_ENGINECFG_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
            return 0;
        case DescriptorType_t::ENGINEHEUR_DESCRIPTOR:
            /*
            DPCT1007:104: Migration of CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
            return 0;
        case DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR:
            /*
            DPCT1007:105: Migration of CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
            return 0;
        case DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR:
            /*
            DPCT1007:106: Migration of CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR;
            return 0;
        case DescriptorType_t::KNOB_CHOICE_DESCRIPTOR:
            /*
            DPCT1007:107: Migration of CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR;
            return 0;
        case DescriptorType_t::KNOB_INFO_DESCRIPTOR:
            /*
            DPCT1007:108: Migration of CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR;
            return 0;
        case DescriptorType_t::LAYOUT_INFO_DESCRIPTOR:
            /*
            DPCT1007:109: Migration of CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            /*
            DPCT1007:110: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            /*
            DPCT1007:111: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            /*
            DPCT1007:112: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR:
            /*
            DPCT1007:113: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR:
            /*
            DPCT1007:114: Migration of CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR:
            /*
            DPCT1007:115: Migration of CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
            return 0;
        case DescriptorType_t::VARIANT_PACK_DESCRIPTOR:
            /*
            DPCT1007:116: Migration of CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
            return 0;
        case DescriptorType_t::TENSOR_DESCRIPTOR:
            /*
            DPCT1007:117: Migration of CUDNN_BACKEND_TENSOR_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_TENSOR_DESCRIPTOR;
            return 0;
        case DescriptorType_t::MATMUL_DESCRIPTOR:
            /*
            DPCT1007:118: Migration of CUDNN_BACKEND_MATMUL_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_MATMUL_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR:
            /*
            DPCT1007:119: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
            /*
            DPCT1007:120: Migration of CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
            return 0;
        case DescriptorType_t::REDUCTION_DESCRIPTOR:
            /*
            DPCT1007:121: Migration of CUDNN_BACKEND_REDUCTION_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_REDUCTION_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR:
            /*
            DPCT1007:122: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            /*
            DPCT1007:123: Migration of CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
            return 0;
        case DescriptorType_t::RESAMPLE_DESCRIPTOR:
            /*
            DPCT1007:124: Migration of CUDNN_BACKEND_RESAMPLE_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_RESAMPLE_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            /*
            DPCT1007:125: Migration of CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR:
#if (CUDNN_VERSION >= 8600)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8600, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:126: Migration of CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
        case DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR:
            /*
            DPCT1007:127: Migration of CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR:
            /*
            DPCT1007:128: Migration of CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR:
            /*
            DPCT1007:129: Migration of CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR:
            /*
            DPCT1007:130: Migration of CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR;
            return 0;
        case DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR:
#if (CUDNN_VERSION >= 8700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:131: Migration of CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

        case DescriptorType_t::RNG_DESCRIPTOR:
#if (CUDNN_VERSION >= 8700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:132: Migration of CUDNN_BACKEND_RNG_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_RNG_DESCRIPTOR;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

        case DescriptorType_t::OPERATION_RNG_DESCRIPTOR:
#if (CUDNN_VERSION >= 8700)
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:133: Migration of CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR is not supported.
            */
            cudnn_mode = CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR;
            return 0;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::ResampleMode_t const mode, cudnnResampleMode_t& cudnn_mode) {
    switch (mode) {
#if (CUDNN_VERSION >= 8600)
        case cudnn_frontend::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING:
            /*
            DPCT1007:134: Migration of CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING is not supported.
            */
            cudnn_mode = CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
            return 0;
        case cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
            /*
            DPCT1007:135: Migration of CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING is not supported.
            */
            cudnn_mode = CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
            return 0;
#else
        case cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
            cudnn_mode = CUDNN_RESAMPLE_AVGPOOL;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#endif
        case cudnn_frontend::ResampleMode_t::BILINEAR:
            /*
            DPCT1007:136: Migration of CUDNN_RESAMPLE_BILINEAR is not supported.
            */
            cudnn_mode = CUDNN_RESAMPLE_BILINEAR;
            return 0;
        case cudnn_frontend::ResampleMode_t::NEAREST:
            /*
            DPCT1007:137: Migration of CUDNN_RESAMPLE_NEAREST is not supported.
            */
            cudnn_mode = CUDNN_RESAMPLE_NEAREST;
            return 0;
        case cudnn_frontend::ResampleMode_t::MAXPOOL:
            /*
            DPCT1007:138: Migration of CUDNN_RESAMPLE_MAXPOOL is not supported.
            */
            cudnn_mode = CUDNN_RESAMPLE_MAXPOOL;
            return 0;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::PaddingMode_t const mode, cudnnPaddingMode_t& cudnn_mode) {
    switch (mode) {
        case cudnn_frontend::PaddingMode_t::ZERO_PAD:
            /*
            DPCT1007:139: Migration of CUDNN_ZERO_PAD is not supported.
            */
            cudnn_mode = CUDNN_ZERO_PAD;
            return 0;
        case cudnn_frontend::PaddingMode_t::NEG_INF_PAD:
            /*
            DPCT1007:140: Migration of CUDNN_NEG_INF_PAD is not supported.
            */
            cudnn_mode = CUDNN_NEG_INF_PAD;
            return 0;
        case cudnn_frontend::PaddingMode_t::EDGE_VAL_PAD:
            /*
            DPCT1007:141: Migration of CUDNN_EDGE_VAL_PAD is not supported.
            */
            cudnn_mode = CUDNN_EDGE_VAL_PAD;
            return 0;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::NormMode_t const mode, cudnnBackendNormMode_t& cudnn_mode) {
    switch (mode) {
        case NormMode_t::LAYER_NORM:
            /*
            DPCT1007:142: Migration of CUDNN_LAYER_NORM is not supported.
            */
            cudnn_mode = CUDNN_LAYER_NORM;
            return 0;
        case NormMode_t::INSTANCE_NORM:
            /*
            DPCT1007:143: Migration of CUDNN_INSTANCE_NORM is not supported.
            */
            cudnn_mode = CUDNN_INSTANCE_NORM;
            return 0;
        case NormMode_t::BATCH_NORM:
            /*
            DPCT1007:144: Migration of CUDNN_BATCH_NORM is not supported.
            */
            cudnn_mode = CUDNN_BATCH_NORM;
            return 0;
        case NormMode_t::GROUP_NORM:
            /*
            DPCT1007:145: Migration of CUDNN_GROUP_NORM is not supported.
            */
            cudnn_mode = CUDNN_GROUP_NORM;
            return 0;

#if (CUDNN_VERSION >= 8906)
        case NormMode_t::RMS_NORM:
            NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8906, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);
            /*
            DPCT1007:146: Migration of CUDNN_RMS_NORM is not supported.
            */
            cudnn_mode = CUDNN_RMS_NORM;
            return 0;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::NormFwdPhase_t const mode, cudnnBackendNormFwdPhase_t& cudnn_mode) {
    switch (mode) {
        case NormFwdPhase_t::INFERENCE:
            /*
            DPCT1007:147: Migration of CUDNN_NORM_FWD_INFERENCE is not supported.
            */
            cudnn_mode = CUDNN_NORM_FWD_INFERENCE;
            return 0;
        case NormFwdPhase_t::TRAINING:
            /*
            DPCT1007:148: Migration of CUDNN_NORM_FWD_TRAINING is not supported.
            */
            cudnn_mode = CUDNN_NORM_FWD_TRAINING;
            return 0;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

// To be deprecated. Only exists as setResampleMode(cudnnPaddingMode_t) requires it.
static inline void
convert_from_cudnn_type(cudnnPaddingMode_t const cudnn_mode, cudnn_frontend::PaddingMode_t& mode) {
    mode = cudnn_frontend::PaddingMode_t::NOT_SET;
    switch (cudnn_mode) {
        /*
        DPCT1007:149: Migration of CUDNN_EDGE_VAL_PAD is not supported.
        */
        case CUDNN_EDGE_VAL_PAD:
            mode = cudnn_frontend::PaddingMode_t::EDGE_VAL_PAD;
            break;
        /*
        DPCT1007:150: Migration of CUDNN_NEG_INF_PAD is not supported.
        */
        case CUDNN_NEG_INF_PAD:
            mode = cudnn_frontend::PaddingMode_t::NEG_INF_PAD;
            break;
        /*
        DPCT1007:151: Migration of CUDNN_ZERO_PAD is not supported.
        */
        case CUDNN_ZERO_PAD:
            mode = cudnn_frontend::PaddingMode_t::ZERO_PAD;
            break;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

// To be deprecated. Only exists as setResampleMode(cudnnResampleMode_t) requires it.
static inline void
convert_from_cudnn_type(cudnnResampleMode_t const cudnn_mode, cudnn_frontend::ResampleMode_t& mode) {
    mode = cudnn_frontend::ResampleMode_t::NOT_SET;
    switch (cudnn_mode) {
#if (CUDNN_VERSION >= 8600)
        /*
        DPCT1007:152: Migration of CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING is not supported.
        */
        case CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING:
            mode = cudnn_frontend::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING;
            break;
        /*
        DPCT1007:153: Migration of CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING is not supported.
        */
        case CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING:
            mode = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
            break;
#else
        case CUDNN_RESAMPLE_AVGPOOL:
            mode = cudnn_frontend::ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
            break;
#endif
        /*
        DPCT1007:154: Migration of CUDNN_RESAMPLE_BILINEAR is not supported.
        */
        case CUDNN_RESAMPLE_BILINEAR:
            mode = cudnn_frontend::ResampleMode_t::BILINEAR;
            break;
        /*
        DPCT1007:155: Migration of CUDNN_RESAMPLE_NEAREST is not supported.
        */
        case CUDNN_RESAMPLE_NEAREST:
            mode = cudnn_frontend::ResampleMode_t::NEAREST;
            break;
        /*
        DPCT1007:156: Migration of CUDNN_RESAMPLE_MAXPOOL is not supported.
        */
        case CUDNN_RESAMPLE_MAXPOOL:
            mode = cudnn_frontend::ResampleMode_t::MAXPOOL;
            break;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

// To be deprecated. Only exists as setNormalizationMode(cudnnBackendNormMode_t) requires it.
static inline void
convert_from_cudnn_type(cudnnBackendNormMode_t const cudnn_mode, cudnn_frontend::NormMode_t& mode) {
    mode = NormMode_t::NOT_SET;
    switch (cudnn_mode) {
        /*
        DPCT1007:157: Migration of CUDNN_LAYER_NORM is not supported.
        */
        case CUDNN_LAYER_NORM:
            mode = NormMode_t::LAYER_NORM;
            break;
        /*
        DPCT1007:158: Migration of CUDNN_INSTANCE_NORM is not supported.
        */
        case CUDNN_INSTANCE_NORM:
            mode = NormMode_t::INSTANCE_NORM;
            break;
        /*
        DPCT1007:159: Migration of CUDNN_BATCH_NORM is not supported.
        */
        case CUDNN_BATCH_NORM:
            mode = NormMode_t::BATCH_NORM;
            break;
        /*
        DPCT1007:160: Migration of CUDNN_GROUP_NORM is not supported.
        */
        case CUDNN_GROUP_NORM:
            mode = NormMode_t::GROUP_NORM;
            break;

#if (CUDNN_VERSION >= 8906)
        /*
        DPCT1007:161: Migration of CUDNN_RMS_NORM is not supported.
        */
        case CUDNN_RMS_NORM:
            mode = NormMode_t::RMS_NORM;
            break;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

// To be deprecated. Only exists as setNormFwdPhase(cudnnBackendNormFwdPhase_t) requires it.
static inline void
convert_from_cudnn_type(cudnnBackendNormFwdPhase_t const cudnn_mode, cudnn_frontend::NormFwdPhase_t& mode) {
    mode = NormFwdPhase_t::NOT_SET;
    switch (cudnn_mode) {
        /*
        DPCT1007:162: Migration of CUDNN_NORM_FWD_INFERENCE is not supported.
        */
        case CUDNN_NORM_FWD_INFERENCE:
            mode = NormFwdPhase_t::INFERENCE;
            break;
        /*
        DPCT1007:163: Migration of CUDNN_NORM_FWD_TRAINING is not supported.
        */
        case CUDNN_NORM_FWD_TRAINING:
            mode = NormFwdPhase_t::TRAINING;
            break;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::TensorReordering_t const mode, cudnnBackendTensorReordering_t& cudnn_mode) {
    switch (mode) {
        case cudnn_frontend::TensorReordering_t::NONE:
            /*
            DPCT1007:164: Migration of CUDNN_TENSOR_REORDERING_NONE is not supported.
            */
            cudnn_mode = CUDNN_TENSOR_REORDERING_NONE;
            return 0;
        case cudnn_frontend::TensorReordering_t::INT8x32:
            /*
            DPCT1007:165: Migration of CUDNN_TENSOR_REORDERING_INT8x32 is not supported.
            */
            cudnn_mode = CUDNN_TENSOR_REORDERING_INT8x32;
            return 0;
        case cudnn_frontend::TensorReordering_t::F16x16:
#if CUDNN_VERSION >= 8800
#if defined NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
            if (get_backend_version() >= 8800) {
                cudnn_mode = CUDNN_TENSOR_REORDERING_F16x16;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            } else if (get_backend_version() >= 8700) {
                cudnn_mode = CUDNN_TENSOR_REORDERING_NONE;
                return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
            } else {
                return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
            }
#endif
            /*
            DPCT1007:166: Migration of CUDNN_TENSOR_REORDERING_F16x16 is not supported.
            */
            cudnn_mode = CUDNN_TENSOR_REORDERING_F16x16;
            return 0;
#elif CUDNN_VERSION >= 8700
            cudnn_mode = CUDNN_TENSOR_REORDERING_NONE;
            return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
#else
            return cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

// To be deprecated. Only exists as setReorderType(cudnnBackendTensorReordering_t) requires it.
static inline void
convert_from_cudnn_type(cudnnBackendTensorReordering_t const cudnn_mode, cudnn_frontend::TensorReordering_t& mode) {
    mode = cudnn_frontend::TensorReordering_t::NONE;
    switch (cudnn_mode) {
        /*
        DPCT1007:167: Migration of CUDNN_TENSOR_REORDERING_INT8x32 is not supported.
        */
        case CUDNN_TENSOR_REORDERING_INT8x32:
            mode = cudnn_frontend::TensorReordering_t::INT8x32;
            break;
#if CUDNN_VERSION >= 8800
        /*
        DPCT1007:168: Migration of CUDNN_TENSOR_REORDERING_F16x16 is not supported.
        */
        case CUDNN_TENSOR_REORDERING_F16x16:
            mode = cudnn_frontend::TensorReordering_t::F16x16;
            break;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            break;
#endif
    }
}

// To be deprecated. Only exists as OperationBuilder_v8(::cudnnBackendDescriptorType_t mode) requires it.
static inline cudnn_frontend::DescriptorType_t
convert_from_cudnn_type(cudnnBackendDescriptorType_t const cudnn_mode) {
    switch (cudnn_mode) {
        /*
        DPCT1007:169: Migration of CUDNN_BACKEND_POINTWISE_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
            return DescriptorType_t::POINTWISE_DESCRIPTOR;
        /*
        DPCT1007:170: Migration of CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
            return DescriptorType_t::CONVOLUTION_DESCRIPTOR;
        /*
        DPCT1007:171: Migration of CUDNN_BACKEND_ENGINE_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_ENGINE_DESCRIPTOR:
            return DescriptorType_t::ENGINE_DESCRIPTOR;
        /*
        DPCT1007:172: Migration of CUDNN_BACKEND_ENGINECFG_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
            return DescriptorType_t::ENGINECFG_DESCRIPTOR;
        /*
        DPCT1007:173: Migration of CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
            return DescriptorType_t::ENGINEHEUR_DESCRIPTOR;
        /*
        DPCT1007:174: Migration of CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
            return DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR;
        /*
        DPCT1007:175: Migration of CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
            return DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR;
        /*
        DPCT1007:176: Migration of CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
            return DescriptorType_t::KNOB_CHOICE_DESCRIPTOR;
        /*
        DPCT1007:177: Migration of CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
            return DescriptorType_t::KNOB_INFO_DESCRIPTOR;
        /*
        DPCT1007:178: Migration of CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
            return DescriptorType_t::LAYOUT_INFO_DESCRIPTOR;
        /*
        DPCT1007:179: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
        /*
        DPCT1007:180: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            return DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
        /*
        DPCT1007:181: Migration of CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            return DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
        /*
        DPCT1007:182: Migration of CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
            return DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR;
        /*
        DPCT1007:183: Migration of CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
            return DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR;
        /*
        DPCT1007:184: Migration of CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
            return DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR;
        /*
        DPCT1007:185: Migration of CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
            return DescriptorType_t::VARIANT_PACK_DESCRIPTOR;
        /*
        DPCT1007:186: Migration of CUDNN_BACKEND_TENSOR_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_TENSOR_DESCRIPTOR:
            return DescriptorType_t::TENSOR_DESCRIPTOR;
        /*
        DPCT1007:187: Migration of CUDNN_BACKEND_MATMUL_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_MATMUL_DESCRIPTOR:
            return DescriptorType_t::MATMUL_DESCRIPTOR;
        /*
        DPCT1007:188: Migration of CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
            return DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR;
        /*
        DPCT1007:189: Migration of CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
            return DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
        /*
        DPCT1007:190: Migration of CUDNN_BACKEND_REDUCTION_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_REDUCTION_DESCRIPTOR:
            return DescriptorType_t::REDUCTION_DESCRIPTOR;
        /*
        DPCT1007:191: Migration of CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
            return DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR;
        /*
        DPCT1007:192: Migration of CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            return DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
        /*
        DPCT1007:193: Migration of CUDNN_BACKEND_RESAMPLE_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_RESAMPLE_DESCRIPTOR:
            return DescriptorType_t::RESAMPLE_DESCRIPTOR;
        /*
        DPCT1007:194: Migration of CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR;
        /*
        DPCT1007:195: Migration of CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
            return DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR;
        /*
        DPCT1007:196: Migration of CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
            return DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR;
        /*
        DPCT1007:197: Migration of CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR;
        /*
        DPCT1007:198: Migration of CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR;
#if (CUDNN_VERSION >= 8600)
        /*
        DPCT1007:199: Migration of CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
            return DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR;
#endif
#if (CUDNN_VERSION >= 8700)
        /*
        DPCT1007:200: Migration of CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
            return DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR;
        /*
        DPCT1007:201: Migration of CUDNN_BACKEND_RNG_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_RNG_DESCRIPTOR:
            return DescriptorType_t::RNG_DESCRIPTOR;
        /*
        DPCT1007:202: Migration of CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR is not supported.
        */
        case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
            return DescriptorType_t::OPERATION_RNG_DESCRIPTOR;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return DescriptorType_t::NOT_SET;
            break;
#endif
    }
    return DescriptorType_t::NOT_SET;
}

// To be deprecated. Only exists as setPointwiseMode(cudnnPointwiseMode_t mode) requires it.
static inline cudnn_frontend::PointwiseMode_t
convert_from_cudnn_type(cudnnPointwiseMode_t const cudnn_mode) {
    switch (cudnn_mode) {
        /*
        DPCT1007:203: Migration of CUDNN_POINTWISE_ADD is not supported.
        */
        case CUDNN_POINTWISE_ADD:
            return PointwiseMode_t::ADD;
        /*
        DPCT1007:204: Migration of CUDNN_POINTWISE_MUL is not supported.
        */
        case CUDNN_POINTWISE_MUL:
            return PointwiseMode_t::MUL;
        /*
        DPCT1007:205: Migration of CUDNN_POINTWISE_SQRT is not supported.
        */
        case CUDNN_POINTWISE_SQRT:
            return PointwiseMode_t::SQRT;
        /*
        DPCT1007:206: Migration of CUDNN_POINTWISE_MAX is not supported.
        */
        case CUDNN_POINTWISE_MAX:
            return PointwiseMode_t::MAX;
        /*
        DPCT1007:207: Migration of CUDNN_POINTWISE_MIN is not supported.
        */
        case CUDNN_POINTWISE_MIN:
            return PointwiseMode_t::MIN;
        /*
        DPCT1007:208: Migration of CUDNN_POINTWISE_RELU_FWD is not supported.
        */
        case CUDNN_POINTWISE_RELU_FWD:
            return PointwiseMode_t::RELU_FWD;
        /*
        DPCT1007:209: Migration of CUDNN_POINTWISE_TANH_FWD is not supported.
        */
        case CUDNN_POINTWISE_TANH_FWD:
            return PointwiseMode_t::TANH_FWD;
        /*
        DPCT1007:210: Migration of CUDNN_POINTWISE_SIGMOID_FWD is not supported.
        */
        case CUDNN_POINTWISE_SIGMOID_FWD:
            return PointwiseMode_t::SIGMOID_FWD;
        /*
        DPCT1007:211: Migration of CUDNN_POINTWISE_ELU_FWD is not supported.
        */
        case CUDNN_POINTWISE_ELU_FWD:
            return PointwiseMode_t::ELU_FWD;
        /*
        DPCT1007:212: Migration of CUDNN_POINTWISE_GELU_FWD is not supported.
        */
        case CUDNN_POINTWISE_GELU_FWD:
            return PointwiseMode_t::GELU_FWD;
        /*
        DPCT1007:213: Migration of CUDNN_POINTWISE_SOFTPLUS_FWD is not supported.
        */
        case CUDNN_POINTWISE_SOFTPLUS_FWD:
            return PointwiseMode_t::SOFTPLUS_FWD;
        /*
        DPCT1007:214: Migration of CUDNN_POINTWISE_SWISH_FWD is not supported.
        */
        case CUDNN_POINTWISE_SWISH_FWD:
            return PointwiseMode_t::SWISH_FWD;
        /*
        DPCT1007:215: Migration of CUDNN_POINTWISE_RELU_BWD is not supported.
        */
        case CUDNN_POINTWISE_RELU_BWD:
            return PointwiseMode_t::RELU_BWD;
        /*
        DPCT1007:216: Migration of CUDNN_POINTWISE_TANH_BWD is not supported.
        */
        case CUDNN_POINTWISE_TANH_BWD:
            return PointwiseMode_t::TANH_BWD;
        /*
        DPCT1007:217: Migration of CUDNN_POINTWISE_SIGMOID_BWD is not supported.
        */
        case CUDNN_POINTWISE_SIGMOID_BWD:
            return PointwiseMode_t::SIGMOID_BWD;
        /*
        DPCT1007:218: Migration of CUDNN_POINTWISE_ELU_BWD is not supported.
        */
        case CUDNN_POINTWISE_ELU_BWD:
            return PointwiseMode_t::ELU_BWD;
        /*
        DPCT1007:219: Migration of CUDNN_POINTWISE_GELU_BWD is not supported.
        */
        case CUDNN_POINTWISE_GELU_BWD:
            return PointwiseMode_t::GELU_BWD;
        /*
        DPCT1007:220: Migration of CUDNN_POINTWISE_SOFTPLUS_BWD is not supported.
        */
        case CUDNN_POINTWISE_SOFTPLUS_BWD:
            return PointwiseMode_t::SOFTPLUS_BWD;
        /*
        DPCT1007:221: Migration of CUDNN_POINTWISE_SWISH_BWD is not supported.
        */
        case CUDNN_POINTWISE_SWISH_BWD:
            return PointwiseMode_t::SWISH_BWD;
        /*
        DPCT1007:222: Migration of CUDNN_POINTWISE_DIV is not supported.
        */
        case CUDNN_POINTWISE_DIV:
            return PointwiseMode_t::DIV;
        /*
        DPCT1007:223: Migration of CUDNN_POINTWISE_ADD_SQUARE is not supported.
        */
        case CUDNN_POINTWISE_ADD_SQUARE:
            return PointwiseMode_t::ADD_SQUARE;
        /*
        DPCT1007:224: Migration of CUDNN_POINTWISE_EXP is not supported.
        */
        case CUDNN_POINTWISE_EXP:
            return PointwiseMode_t::EXP;
        /*
        DPCT1007:225: Migration of CUDNN_POINTWISE_SUB is not supported.
        */
        case CUDNN_POINTWISE_SUB:
            return PointwiseMode_t::SUB;
        /*
        DPCT1007:226: Migration of CUDNN_POINTWISE_CMP_EQ is not supported.
        */
        case CUDNN_POINTWISE_CMP_EQ:
            return PointwiseMode_t::CMP_EQ;
        /*
        DPCT1007:227: Migration of CUDNN_POINTWISE_CMP_NEQ is not supported.
        */
        case CUDNN_POINTWISE_CMP_NEQ:
            return PointwiseMode_t::CMP_NEQ;
        /*
        DPCT1007:228: Migration of CUDNN_POINTWISE_CMP_GT is not supported.
        */
        case CUDNN_POINTWISE_CMP_GT:
            return PointwiseMode_t::CMP_GT;
        /*
        DPCT1007:229: Migration of CUDNN_POINTWISE_CMP_GE is not supported.
        */
        case CUDNN_POINTWISE_CMP_GE:
            return PointwiseMode_t::CMP_GE;
        /*
        DPCT1007:230: Migration of CUDNN_POINTWISE_CMP_LT is not supported.
        */
        case CUDNN_POINTWISE_CMP_LT:
            return PointwiseMode_t::CMP_LT;
        /*
        DPCT1007:231: Migration of CUDNN_POINTWISE_CMP_LE is not supported.
        */
        case CUDNN_POINTWISE_CMP_LE:
            return PointwiseMode_t::CMP_LE;
        /*
        DPCT1007:232: Migration of CUDNN_POINTWISE_LOGICAL_AND is not supported.
        */
        case CUDNN_POINTWISE_LOGICAL_AND:
            return PointwiseMode_t::LOGICAL_AND;
        /*
        DPCT1007:233: Migration of CUDNN_POINTWISE_LOGICAL_OR is not supported.
        */
        case CUDNN_POINTWISE_LOGICAL_OR:
            return PointwiseMode_t::LOGICAL_OR;
        /*
        DPCT1007:234: Migration of CUDNN_POINTWISE_LOGICAL_NOT is not supported.
        */
        case CUDNN_POINTWISE_LOGICAL_NOT:
            return PointwiseMode_t::LOGICAL_NOT;
        /*
        DPCT1007:235: Migration of CUDNN_POINTWISE_LOG is not supported.
        */
        case CUDNN_POINTWISE_LOG:
            return PointwiseMode_t::LOG;
        /*
        DPCT1007:236: Migration of CUDNN_POINTWISE_NEG is not supported.
        */
        case CUDNN_POINTWISE_NEG:
            return PointwiseMode_t::NEG;
        /*
        DPCT1007:237: Migration of CUDNN_POINTWISE_MOD is not supported.
        */
        case CUDNN_POINTWISE_MOD:
            return PointwiseMode_t::MOD;
        /*
        DPCT1007:238: Migration of CUDNN_POINTWISE_POW is not supported.
        */
        case CUDNN_POINTWISE_POW:
            return PointwiseMode_t::POW;
        /*
        DPCT1007:239: Migration of CUDNN_POINTWISE_ABS is not supported.
        */
        case CUDNN_POINTWISE_ABS:
            return PointwiseMode_t::ABS;
        /*
        DPCT1007:240: Migration of CUDNN_POINTWISE_CEIL is not supported.
        */
        case CUDNN_POINTWISE_CEIL:
            return PointwiseMode_t::CEIL;
        /*
        DPCT1007:241: Migration of CUDNN_POINTWISE_COS is not supported.
        */
        case CUDNN_POINTWISE_COS:
            return PointwiseMode_t::COS;
        /*
        DPCT1007:242: Migration of CUDNN_POINTWISE_FLOOR is not supported.
        */
        case CUDNN_POINTWISE_FLOOR:
            return PointwiseMode_t::FLOOR;
        /*
        DPCT1007:243: Migration of CUDNN_POINTWISE_RSQRT is not supported.
        */
        case CUDNN_POINTWISE_RSQRT:
            return PointwiseMode_t::RSQRT;
        /*
        DPCT1007:244: Migration of CUDNN_POINTWISE_SIN is not supported.
        */
        case CUDNN_POINTWISE_SIN:
            return PointwiseMode_t::SIN;
        /*
        DPCT1007:245: Migration of CUDNN_POINTWISE_TAN is not supported.
        */
        case CUDNN_POINTWISE_TAN:
            return PointwiseMode_t::TAN;
        /*
        DPCT1007:246: Migration of CUDNN_POINTWISE_GEN_INDEX is not supported.
        */
        case CUDNN_POINTWISE_GEN_INDEX:
            return PointwiseMode_t::GEN_INDEX;
        /*
        DPCT1007:247: Migration of CUDNN_POINTWISE_BINARY_SELECT is not supported.
        */
        case CUDNN_POINTWISE_BINARY_SELECT:
            return PointwiseMode_t::BINARY_SELECT;
        /*
        DPCT1007:248: Migration of CUDNN_POINTWISE_ERF is not supported.
        */
        case CUDNN_POINTWISE_ERF:
            return PointwiseMode_t::ERF;
        /*
        DPCT1007:249: Migration of CUDNN_POINTWISE_IDENTITY is not supported.
        */
        case CUDNN_POINTWISE_IDENTITY:
            return PointwiseMode_t::IDENTITY;
        /*
        DPCT1007:250: Migration of CUDNN_POINTWISE_GELU_APPROX_TANH_BWD is not supported.
        */
        case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
            return PointwiseMode_t::GELU_APPROX_TANH_BWD;
        /*
        DPCT1007:251: Migration of CUDNN_POINTWISE_GELU_APPROX_TANH_FWD is not supported.
        */
        case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
            return PointwiseMode_t::GELU_APPROX_TANH_FWD;
#if (CUDNN_VERSION >= 8900)
        /*
        DPCT1007:252: Migration of CUDNN_POINTWISE_RECIPROCAL is not supported.
        */
        case CUDNN_POINTWISE_RECIPROCAL:
            return PointwiseMode_t::RECIPROCAL;
#endif

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return PointwiseMode_t::NOT_SET;
#endif
    }
    return PointwiseMode_t::NOT_SET;
}

// To be deprecated. Only exists as setDataType(cudnnDataType_t mode) requires it.
static inline cudnn_frontend::DataType_t
convert_from_cudnn_type(dpct::library_data_t const cudnn_mode) {
    switch (cudnn_mode) {
        case dpct::library_data_t::real_float:
            return DataType_t::FLOAT;
        case dpct::library_data_t::real_double:
            return DataType_t::DOUBLE;
        case dpct::library_data_t::real_half:
            return DataType_t::HALF;
        case dpct::library_data_t::real_int8:
            return DataType_t::INT8;
        case dpct::library_data_t::real_int32:
            return DataType_t::INT32;
        case dpct::library_data_t::real_int8_4:
            return DataType_t::INT8x4;
        case dpct::library_data_t::real_uint8:
            return DataType_t::UINT8;
        case dpct::library_data_t::real_uint8_4:
            return DataType_t::UINT8x4;
        case dpct::library_data_t::real_int8_32:
            return DataType_t::INT8x32;
        case dpct::library_data_t::real_bfloat16:
            return DataType_t::BFLOAT16;
        /*
        DPCT1007:253: Migration of CUDNN_DATA_INT64 is not supported.
        */
        case CUDNN_DATA_INT64:
            return DataType_t::INT64;
        /*
        DPCT1007:254: Migration of CUDNN_DATA_BOOLEAN is not supported.
        */
        case CUDNN_DATA_BOOLEAN:
            return DataType_t::BOOLEAN;
#if (CUDNN_VERSION >= 8600)
        /*
        DPCT1007:255: Migration of CUDNN_DATA_FP8_E4M3 is not supported.
        */
        case CUDNN_DATA_FP8_E4M3:
            return DataType_t::FP8_E4M3;
        /*
        DPCT1007:256: Migration of CUDNN_DATA_FP8_E5M2 is not supported.
        */
        case CUDNN_DATA_FP8_E5M2:
            return DataType_t::FP8_E5M2;
#endif
#if (CUDNN_VERSION >= 8700)
        /*
        DPCT1007:257: Migration of CUDNN_DATA_FAST_FLOAT_FOR_FP8 is not supported.
        */
        case CUDNN_DATA_FAST_FLOAT_FOR_FP8:
            return DataType_t::FAST_FLOAT_FOR_FP8;
#endif
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return DataType_t::NOT_SET;
#endif
    }
    return DataType_t::NOT_SET;
}

// To be deprecated. Only exists as setReductionOp(cudnnReduceTensorOp_t mode) requires it.
static inline cudnn_frontend::ReductionMode_t
convert_from_cudnn_type(dpct::dnnl::reduction_op const cudnn_mode) {
    switch (cudnn_mode) {
        case dpct::dnnl::reduction_op::sum:
            return ReductionMode_t::ADD;
        case dpct::dnnl::reduction_op::mul:
            return ReductionMode_t::MUL;
        case dpct::dnnl::reduction_op::min:
            return ReductionMode_t::MIN;
        case dpct::dnnl::reduction_op::max:
            return ReductionMode_t::MAX;
        case dpct::dnnl::reduction_op::amax:
            return ReductionMode_t::AMAX;
        case dpct::dnnl::reduction_op::mean:
            return ReductionMode_t::AVG;
        case dpct::dnnl::reduction_op::norm1:
            return ReductionMode_t::NORM1;
        case dpct::dnnl::reduction_op::norm2:
            return ReductionMode_t::NORM2;
        case dpct::dnnl::reduction_op::mul_no_zeros:
            return ReductionMode_t::MUL_NO_ZEROS;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return ReductionMode_t::NOT_SET;
#endif
    }
    return ReductionMode_t::NOT_SET;
}

#if (CUDNN_VERSION >= 8700)
static inline dpct::err1
convert_to_cudnn_type(cudnn_frontend::RngDistribution_t const mode, cudnnRngDistribution_t& cudnn_mode) {
    NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(8700, cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE);

    switch (mode) {
        case RngDistribution_t::BERNOULLI:
            /*
            DPCT1007:258: Migration of CUDNN_RNG_DISTRIBUTION_BERNOULLI is not supported.
            */
            cudnn_mode = CUDNN_RNG_DISTRIBUTION_BERNOULLI;
            return 0;
        case RngDistribution_t::UNIFORM:
            /*
            DPCT1007:259: Migration of CUDNN_RNG_DISTRIBUTION_UNIFORM is not supported.
            */
            cudnn_mode = CUDNN_RNG_DISTRIBUTION_UNIFORM;
            return 0;
        case RngDistribution_t::NORMAL:
            /*
            DPCT1007:260: Migration of CUDNN_RNG_DISTRIBUTION_NORMAL is not supported.
            */
            cudnn_mode = CUDNN_RNG_DISTRIBUTION_NORMAL;
            return 0;

#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return 2001;
#endif
    }
    return 2001;
}

// To be deprecated. Only exists as setRngDistribution(cudnnRngDistribution_t mode) requires it.
static inline cudnn_frontend::RngDistribution_t
convert_from_cudnn_type(cudnnRngDistribution_t const cudnn_mode) {
    switch (cudnn_mode) {
        /*
        DPCT1007:261: Migration of CUDNN_RNG_DISTRIBUTION_BERNOULLI is not supported.
        */
        case CUDNN_RNG_DISTRIBUTION_BERNOULLI:
            return RngDistribution_t::BERNOULLI;
        /*
        DPCT1007:262: Migration of CUDNN_RNG_DISTRIBUTION_UNIFORM is not supported.
        */
        case CUDNN_RNG_DISTRIBUTION_UNIFORM:
            return RngDistribution_t::UNIFORM;
        /*
        DPCT1007:263: Migration of CUDNN_RNG_DISTRIBUTION_NORMAL is not supported.
        */
        case CUDNN_RNG_DISTRIBUTION_NORMAL:
            return RngDistribution_t::NORMAL;
#ifndef NO_DEFAULT_IN_SWITCH
        default:
            return RngDistribution_t::NOT_SET;
#endif
    }
    return RngDistribution_t::NOT_SET;
}
#endif

}  // namespace detail

}  // namespace cudnn_frontend
