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
    \brief Defines basic properties needed by CTA-level batched GEMV assuming expectations about data
      layout of the global memory fragments, data types, and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting SIMT instructions.
*/

#pragma once

#include "cutlass_3_5/cutlass.h"
#include "cutlass_3_5/array.h"
#include "cutlass_3_5/numeric_types.h"
#include "cutlass_3_5/matrix_shape.h"

#include "cutlass_3_5/layout/matrix.h"

#include "cutlass_3_5/platform/platform.h"

#include "cutlass_3_5/gemm/gemm.h"
#include "cutlass_3_5/gemm/thread/mma.h"

#include "cutlass_3_5/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass_3_5/transform/pitch_linear_thread_map.h"

#include "cutlass_3_5/gemm/threadblock/gemv.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass_3_5 {
namespace gemm {
namespace threadblock {

/// Template defininng default vector-matrix multiply operators inferred from threadblock tile size,
/// global memory data layout.
template <
  typename Shape_,            /// Shape of the threadblock vector-matrix multiply operator
  typename ThreadShape_,      /// Shape of per-thread vector-matrix multiply operator
  typename ElementA_,         /// Element data type of A operand
  typename LayoutA_,          /// Layout of operand A
  typename ElementB_,         /// Element data type of B operand
  typename LayoutB_,          /// Layout of operand B
  typename ElementC_,         /// Data type of accumulator
  typename LayoutC_           /// Layout of accumulator
>
struct DefaultGemvCore {

  using Shape = Shape_;
  using ThreadShape = ThreadShape_;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;
  using LayoutC = LayoutC_;
  
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;

  static int const kThreadsPerN = Shape::kN / ThreadShape::kN;

  using IteratorPolicyA = typename platform::conditional<
                            platform::is_same<LayoutA, layout::RowMajor>::value,
                            cutlass_3_5::transform::PitchLinearTilePolicyStripminedThreadContiguous<
                              layout::PitchLinearShape<Shape::kK, Shape::kM>, 1, ThreadShape::kK>,
                            cutlass_3_5::transform::PitchLinearTilePolicyStripminedThreadStrided<
                              layout::PitchLinearShape<Shape::kM, Shape::kK>, 1, ThreadShape::kM>>::type;

  using IteratorA = cutlass_3_5::transform::threadblock::PredicatedTileIterator<
                          cutlass_3_5::MatrixShape<Shape::kM, Shape::kK>, ElementA, LayoutA, 1, IteratorPolicyA>;

  using IteratorPolicyB = typename platform::conditional<
                            platform::is_same<LayoutB, layout::RowMajor>::value,
                            cutlass_3_5::transform::PitchLinearTilePolicyStripminedThreadContiguous<
                              layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreadsPerN, ThreadShape::kN>,
                            cutlass_3_5::transform::PitchLinearTilePolicyStripminedThreadStrided<
                              layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreadsPerN, ThreadShape::kK>>::type;

  using IteratorB = cutlass_3_5::transform::threadblock::PredicatedTileIterator<
                            cutlass_3_5::MatrixShape<Shape::kK, Shape::kN>, ElementB, LayoutB, 0, IteratorPolicyB>;

  using IteratorPolicyC = typename platform::conditional<
                            platform::is_same<LayoutC, layout::RowMajor>::value,
                            cutlass_3_5::transform::PitchLinearTilePolicyStripminedThreadContiguous<
                              layout::PitchLinearShape<Shape::kN, Shape::kM>, kThreadsPerN, ThreadShape::kN>,
                            cutlass_3_5::transform::PitchLinearTilePolicyStripminedThreadStrided<
                              layout::PitchLinearShape<Shape::kM, Shape::kN>, kThreadsPerN, ThreadShape::kM>>::type;

  using IteratorC = cutlass_3_5::transform::threadblock::PredicatedTileIterator<
                             cutlass_3_5::MatrixShape<Shape::kM, Shape::kN>, ElementC, LayoutC, 0, IteratorPolicyC>;

  using MmaSimtOp = typename cutlass_3_5::gemm::thread::Mma<
    cutlass_3_5::gemm::GemmShape<ThreadShape::kM, ThreadShape::kN, Shape::kK>,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC>;

  using Operator = MmaSimtOp;

  // Assertions for correctness
  static_assert((Shape::kM == 1), "M=1 is required for GEMV");
  
  static_assert((ThreadShape::kM == 1), "M=1 is required for GEMV");

  static_assert(Shape::kK % ThreadShape::kK == 0, "Shape::K must be a multiple of ThreadShape::K");

  static_assert(((ThreadShape::kK == 1) ||
                (ThreadShape::kK == 2) || 
                (ThreadShape::kK == 4) ||
                (ThreadShape::kK == 8) ||
                (ThreadShape::kK == 16) ||
                (ThreadShape::kK == 32)
               ),
              "ThreadShape::K must be a 1, 2, 4, 8, 16 or 32");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass_3_5
