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
  \brief Epilogue for threadblock scoped GEMMs using SIMT.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass_3_5/cutlass.h"
#include "cutlass_3_5/numeric_types.h"
#include "cutlass_3_5/array.h"

#include "cutlass_3_5/arch/mma.h"

#include "cutlass_3_5/gemm/gemm.h"
#include "cutlass_3_5/gemm/warp/mma.h"

#include "cutlass_3_5/epilogue/thread/linear_combination.h"
#include "cutlass_3_5/epilogue/thread/linear_combination_clamp.h"
#include "cutlass_3_5/epilogue/thread/linear_combination_relu.h"
#include "cutlass_3_5/epilogue/thread/linear_combination_gelu.h"
#include "cutlass_3_5/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass_3_5/epilogue/thread/linear_combination_planar_complex.h"
#include "cutlass_3_5/epilogue/thread/conversion_op.h"
#include "cutlass_3_5/epilogue/thread/reduction_op.h"

#include "cutlass_3_5/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "cutlass_3_5/epilogue/warp/fragment_iterator_simt.h"
#include "cutlass_3_5/epilogue/warp/tile_iterator_simt.h"
#include "cutlass_3_5/epilogue/threadblock/default_thread_map_simt.h"
#include "cutlass_3_5/transform/pitch_linear_thread_map.h"

#include "cutlass_3_5/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass_3_5/epilogue/threadblock/predicated_tile_iterator_conv.h"
#include "cutlass_3_5/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h"
#include "cutlass_3_5/epilogue/threadblock/predicated_tile_iterator_affine.h"
#include "cutlass_3_5/epilogue/threadblock/predicated_tile_iterator_direct_conv.h" 
#include "cutlass_3_5/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass_3_5/epilogue/threadblock/shared_load_iterator_pitch_linear.h"
#include "cutlass_3_5/epilogue/threadblock/epilogue.h"
#include "cutlass_3_5/epilogue/threadblock/epilogue_depthwise.h"

#include "cutlass_3_5/layout/permute.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_3_5 {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <
  typename Shape_,
  typename WarpMmaSimt_,
  typename OutputOp_,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute,
  conv::StrideSupport StrideSupport = conv::StrideSupport::kUnity,
  int Rank = 4
>
struct DefaultEpilogueSimt {

  using Shape = Shape_;
  using WarpMmaSimt = WarpMmaSimt_;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaSimt::LayoutC;
  using ElementAccumulator = typename WarpMmaSimt::ElementC;
  static conv::StrideSupport const kStrideSupport = StrideSupport;
  static int const kRank = Rank;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass_3_5::epilogue::threadblock::DefaultThreadMapSimt<
    Shape,
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::Policy,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  static bool const UseCUDAStore = platform::is_same<ElementOutput, double>::value;

  using PackedOutputTileIterator = cutlass_3_5::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore
  >;

  using StridedOutputTileIterator = cutlass_3_5::epilogue::threadblock::PredicatedTileIteratorConv<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore,
    kRank
  >;

  using OutputTileIterator = typename platform::conditional<StrideSupport == cutlass_3_5::conv::StrideSupport::kUnity,
                                                            PackedOutputTileIterator,
                                                            StridedOutputTileIterator>::type;

  using AccumulatorFragmentIterator = cutlass_3_5::epilogue::warp::FragmentIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using WarpTileIterator = cutlass_3_5::epilogue::warp::TileIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    ElementAccumulator,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using SharedLoadIterator = cutlass_3_5::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass_3_5::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaSimt,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <
  typename Shape_,
  typename WarpMmaSimt_,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueSimtStridedDgrad {

  using Shape = Shape_;
  using WarpMmaSimt = WarpMmaSimt_;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaSimt::LayoutC;
  using ElementAccumulator = typename WarpMmaSimt::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass_3_5::epilogue::threadblock::DefaultThreadMapSimt<
    Shape,
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::Policy,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = cutlass_3_5::epilogue::threadblock::PredicatedTileIteratorStridedDgrad<
    OutputTileThreadMap,
    ElementOutput
  >;

  using AccumulatorFragmentIterator = cutlass_3_5::epilogue::warp::FragmentIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using WarpTileIterator = cutlass_3_5::epilogue::warp::TileIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    ElementAccumulator,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using SharedLoadIterator = cutlass_3_5::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass_3_5::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaSimt,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <
  int Rank,
  typename Shape_,
  typename WarpMmaSimt_,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueSimtAffineRankN {

  using Shape = Shape_;
  using WarpMmaSimt = WarpMmaSimt_;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaSimt::LayoutC;
  using ElementAccumulator = typename WarpMmaSimt::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass_3_5::epilogue::threadblock::DefaultThreadMapSimt<
    Shape,
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::Policy,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = cutlass_3_5::epilogue::threadblock::PredicatedTileIteratorAffineRankN<
    OutputTileThreadMap,
    ElementOutput,
    Rank
  >;

  using AccumulatorFragmentIterator = cutlass_3_5::epilogue::warp::FragmentIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using WarpTileIterator = cutlass_3_5::epilogue::warp::TileIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    ElementAccumulator,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using SharedLoadIterator = cutlass_3_5::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass_3_5::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaSimt,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <typename Shape_,        // ThreadBlock Shape
          typename WarpMmaSimt_,  // mma_depthwise_simt
          typename OutputOp_,
          int ElementsPerAccess_,
          typename ThreadOutputShape_ = cutlass_3_5::conv::TensorNHWCShape<1, 1, 1, 1>,
          typename ThreadBlockOutputShape_ = cutlass_3_5::conv::TensorNHWCShape<1, 1, 1, 1> >
struct DefaultDirectConvEpilogueSimt {
  using Shape = Shape_;
  using WarpMmaSimt = WarpMmaSimt_;
  using WarpShape = typename WarpMmaSimt::Shape;
  using OutputOp = OutputOp_;
  using ThreadOutputShape = ThreadOutputShape_;
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;
  static int const kElementsPerAccess = ElementsPerAccess_;


  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaSimt::LayoutC;
  using ElementAccumulator = typename WarpMmaSimt::ElementC;

  /// Number of threads total
  using WarpCount = gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN
  >;

  static int const kWarpSize = cutlass_3_5::gemm::warp::WarpSize<arch::OpClassSimt>::value;

  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Thread map
  //
  
  using OutputTileThreadMap = cutlass_3_5::transform::PitchLinearStripminedThreadMap<
    layout::PitchLinearShape<ThreadBlockOutputShape::kC, ThreadBlockOutputShape::kNHW>,
    kThreads,
    kElementsPerAccess
  >;


  using OutputTileIterator = cutlass_3_5::epilogue::threadblock::PredicatedTileIteratorDirectConv<
    OutputTileThreadMap,
    ElementOutput,
    ThreadOutputShape,
    ThreadBlockOutputShape 
  >;

  using AccumulatorFragmentIterator = cutlass_3_5::epilogue::warp::FragmentIteratorSimt<
    typename WarpMmaSimt::Shape,
    typename WarpMmaSimt::ThreadMma,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;
  
  using WarpTileIterator = cutlass_3_5::epilogue::warp::TileIteratorSimtDirect2dConv<
    typename WarpMmaSimt::Shape,
    ThreadOutputShape,
    ThreadBlockOutputShape,
    typename WarpMmaSimt::ThreadMma,
    ElementAccumulator,
    layout::RowMajor,
    typename WarpMmaSimt::Policy
  >;

  using SharedLoadIterator = cutlass_3_5::epilogue::threadblock::SharedLoadIteratorPitchLinear<
    OutputTileThreadMap,
    ElementAccumulator
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;
  //
  // Define the epilogue
  //
  using Epilogue = cutlass_3_5::epilogue::threadblock::EpilogueDepthwise<
    Shape,
    ThreadOutputShape,
    ThreadBlockOutputShape,
    WarpMmaSimt,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass_3_5

/////////////////////////////////////////////////////////////////////////////////////////////////
