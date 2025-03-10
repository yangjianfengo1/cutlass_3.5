/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief
*/

#pragma once

#include "cutlass_3_5/cutlass.h"
#include "cutlass_3_5/numeric_types.h"
#include "cutlass_3_5/arch/arch.h"
#include "cutlass_3_5/device_kernel.h"

#include "cutlass_3_5/gemm/gemm.h"
#include "cutlass_3_5/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass_3_5/gemm/kernel/gemm_universal.h"

#include "cutlass_3_5/gemm/kernel/default_gemm_universal.h"
#include "cutlass_3_5/gemm/device/default_gemm_configuration.h"
#include "cutlass_3_5/gemm/device/gemm_universal_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_3_5 {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemvKernel_>
class Gemv {
public:

  using GemvKernel = GemvKernel_;


  using ElementA = typename GemvKernel::ElementA;
  using LayoutA  = typename GemvKernel::LayoutA;
  using ElementB = typename GemvKernel::ElementB;
  using ElementC = typename GemvKernel::ElementC;

  using ElementAccumulator = typename GemvKernel::ElementAccumulator;
  using EpilogueOutputOp = typename GemvKernel::EpilogueOutputOp;

  static ComplexTransform const kTransformA = GemvKernel::kTransformA;
  static ComplexTransform const kTransformB = GemvKernel::kTransformB;

  static int const kThreadCount = GemvKernel::kThreadCount;
  static int const kThreadsPerRow = GemvKernel::kThreadsPerRow;

  using Arguments = typename GemvKernel::Arguments;
  using Params = typename GemvKernel::Params;

private:

  Params params_;

public:

  /// Constructs the Gemv.
  Gemv() { }

  /// Determines whether the Gemv can execute_3_5 the given problem.
  static Status can_implement(Arguments const &args) {

    return GemvKernel::can_implement(args);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    return 0;
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args, dim3 const &block) { 
    if(platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      return dim3((args.problem_size.row() + (block.x - 1)) / block.x, 1, args.batch_count % 65536);
    }
    else {
      return dim3((args.problem_size.row() + (block.y - 1)) / block.y, 1, args.batch_count % 65536);
    }
  }

  /// Computes the block shape
  static dim3 get_block_shape() { 
    if(platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      return dim3(kThreadCount, 1, 1);
    }
    else {
      return dim3(kThreadsPerRow, kThreadCount / kThreadsPerRow, 1);
    }
  }

  /// Initializes Gemv state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
    params_ = Params(args);
    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    return params_.update(args);    
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    dim3 block = get_block_shape();
    dim3 grid = get_grid_shape(params_, block);

    int smem_size = int(sizeof(typename GemvKernel::SharedStorage));
    
    // Launch
    cutlass_3_5::Kernel<GemvKernel><<<grid, block, smem_size, stream>>>(params_);

    //
    // Query for errors
    //
    cudaError_t result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass_3_5

////////////////////////////////////////////////////////////////////////////////
