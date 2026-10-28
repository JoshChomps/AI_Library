#pragma once

/**
 * @file kinesis.hpp
 * @brief Master include header for the Kinesis Low-Level AI Engine.
 */

// Core Memory & Tensor Architecture
#include "kinesis/core/types.hpp"
#include "kinesis/core/virtual_memory.hpp"
#include "kinesis/core/allocator.hpp"
#include "kinesis/core/arena_allocator.hpp"
#include "kinesis/core/slab_allocator.hpp"
#include "kinesis/core/memory_sandbox.hpp"
#include "kinesis/core/hardware_info.hpp"
#include "kinesis/core/shape.hpp"
#include "kinesis/core/tensor_view.hpp"
#include "kinesis/core/tensor.hpp"

// High-Performance Compute Kernels
#include "kinesis/compute/simd.hpp"
#include "kinesis/compute/elementwise.hpp"
#include "kinesis/compute/gemm.hpp"
#include "kinesis/compute/gemm_pack.hpp"

// Automatic Differentiation & Dual Numbers
#include "kinesis/autograd/dual.hpp"

