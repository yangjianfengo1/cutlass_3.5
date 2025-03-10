/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "cutlass_3_5/cutlass.h"
#include "cutlass_3_5/numeric_conversion.h"
#include "cutlass_3_5/gemm/gemm.h"
#include "cutlass_3_5/detail/dependent_false.hpp"
#include "cutlass_3_5/gemm/dispatch_policy.hpp"
#include "cutlass_3_5/numeric_types.h"
#include "cutlass_3_5/detail/layout.hpp"
#include "cutlass_3_5/pipeline/pipeline.hpp"
#include "cutlass_3_5/transform/collective/sm90_wgmma_transpose.hpp"
#include "cutlass_3_5/trace.h"
#include "cutlass_3_5/detail/collective.hpp"

#include "cute_3_5/arch/cluster_sm90.hpp"
#include "cute_3_5/arch/copy_sm90.hpp"
#include "cute_3_5/algorithm/functional.hpp"
#include "cute_3_5/atom/mma_atom.hpp"
#include "cute_3_5/atom/copy_traits_sm90_tma.hpp"
#include "cute_3_5/algorithm/gemm.hpp"
#include "cute_3_5/tensor_predicate.hpp"
#include "cute_3_5/numeric/arithmetic_tuple.hpp"
#include "cutlass_3_5/pipeline/pipeline.hpp"
#include "cutlass_3_5/trace.h"
#include "cutlass_3_5/detail/collective.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_3_5::gemm::collective {
using namespace cute_3_5;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop that source A operand from registers
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementAOptionalTuple,
  class StrideA_,
  class ElementBOptionalTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
private:
  template <class PointerType>
  static constexpr auto
  get_logical_ptr(PointerType const* ptr) {
    if constexpr (cute_3_5::sizeof_bits_v<PointerType> < 8) {
      return subbyte_iterator<PointerType const>(ptr);
    }
    else {  
      return ptr;
    }
  }

  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };

  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;

  static_assert(cute_3_5::is_tuple<ElementAOptionalTuple>::value ^ cute_3_5::is_tuple<ElementBOptionalTuple>::value, 
    "Either A OR B must be a tuple. It must take the from {ElementOperand, [ElementScale], [ElementZero]}. Inputs in [] are optional.");

  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute_3_5::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute_3_5::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute_3_5::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute_3_5::conditional_t<cute_3_5::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero = cute_3_5::conditional_t<cute_3_5::is_void_v<ElementZero>, float, ElementZero>;

  using StrideA = StrideA_;
  using StrideB = StrideB_;
  // These are always MN major
  using StrideScale = cute_3_5::Stride<cute_3_5::Int<1>, int64_t, int64_t>;
  // For cases where we can't have a void scale, we can use this to allow the code to compile when the scale is void.
  using NonVoidStrideScale = cute_3_5::conditional_t<cute_3_5::is_void_v<StrideScale>, cute_3_5::Stride<_1, int64_t, int64_t>, StrideScale>;

  static_assert((IsATransformed && cutlass_3_5::gemm::detail::is_k_major<StrideA>()) || 
                (!IsATransformed && cutlass_3_5::gemm::detail::is_k_major<StrideB>()),
                "The transformed type must be K-major.");

  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||
                (!IsATransformed && (sizeof(ElementA) == 2)) ||
                (cutlass_3_5::gemm::detail::is_k_major<StrideA>() && 
                 cutlass_3_5::gemm::detail::is_k_major<StrideB>()), 
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass_3_5::gemm::detail::is_mn_major<NonVoidStrideScale>(), 
    "Scale must be MN major [Col Major if A is scaled, Row Major if B is scaled].");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));

  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = cute_3_5::SM90_TMA_LOAD;

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  // Scale layout atom set after swapping.

  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using SmemCopyAtomScale = Copy_Atom<cute_3_5::DefaultCopy, NonVoidElementScale>;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = !IsATransformed;
  using InternalSmemLayoutAtomA = cute_3_5::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using InternalSmemLayoutAtomB = cute_3_5::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using InternalSmemCopyAtomA   = cute_3_5::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using InternalSmemCopyAtomB   = cute_3_5::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute_3_5::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute_3_5::is_same_v<float, ElementB>;
  using ConvertedElementA = cute_3_5::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB = cute_3_5::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealInternalElementA = cute_3_5::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealInternalElementB = cute_3_5::conditional_t<!SwapAB, ElementB, ElementA>;
  using InternalElementA = cute_3_5::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using InternalElementB = cute_3_5::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;
  using InternalStrideA  = cute_3_5::conditional_t<!SwapAB, StrideA, StrideB>;
  using InternalStrideB  = cute_3_5::conditional_t<!SwapAB, StrideB, StrideA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using InternalTransformA  = cute_3_5::conditional_t<!SwapAB, TransformA, TransformB>;
  using InternalTransformB  = cute_3_5::conditional_t<!SwapAB, TransformB, TransformA>;

  static constexpr int IsSubbyteA = cute_3_5::sizeof_bits_v<InternalElementA> < 8;
  using TmaElementA = cute_3_5::conditional_t<IsSubbyteA, uint8_t, InternalElementA>;

  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass_3_5::PipelineTmaAsync<
                             DispatchPolicy::Stages>;
  using PipelineState = cutlass_3_5::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;

  using SmemLayoutAtomScale = Layout<Shape<decltype(cute_3_5::shape<0>(InternalSmemLayoutAtomA{})), cute_3_5::Int<1>>>;
  using ScaleTileShape = decltype(make_shape(shape<0>(TileShape{}), shape<1>(SmemLayoutAtomScale{})));

  static_assert(cute_3_5::rank(InternalSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute_3_5::rank(InternalSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(InternalSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScale{}) == 2, "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0, "SmemLayoutAtomScale must evenly divide tile k shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      InternalSmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute_3_5::conditional_t< ::cutlass_3_5::gemm::detail::is_major<0,InternalStrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      InternalSmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute_3_5::conditional_t< ::cutlass_3_5::gemm::detail::is_major<0,InternalStrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
    
  // It is assumed that the scales and zero-points share the same smem layout
  using SmemLayoutScale = decltype(tile_to_shape(
    SmemLayoutAtomScale{}, 
    make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}), Int<Stages>{}),
    cute_3_5::conditional_t< ::cutlass_3_5::gemm::detail::is_major<0,NonVoidStrideScale>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute_3_5::is_base_of<cute_3_5::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                    cute_3_5::is_base_of<cute_3_5::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");
  static_assert(cute_3_5::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute_3_5::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute_3_5::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute_3_5::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // To relax them, we need to handle loading more than 1 row of scales for every main loop iteration.
  // We must also handle updating the pipeline transaction bytes on the fly.
  // NOTE: Deleting this assertion without required changes will cause the code to hang.
  static_assert(size<1>(SmemLayoutAtomScale{}) == 1, "size<1>(SmemLayoutAtomScale) must be 1.");

private:
  static constexpr ConversionMode 
  get_conversion_mode() {
    if constexpr (cute_3_5::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } 
    else if constexpr (cute_3_5::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;

  static constexpr auto
  elements_per_smem_scale() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    } 
    else if constexpr (ModeHasScales) {
      return cute_3_5::cosize_v<SmemLayoutScale>;
    } 
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
    }
  }

  static constexpr auto
  elements_per_smem_zero() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert ||
                  KernelConversionMode == ConversionMode::ConvertAndScale ) {
      return 0;
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      return cute_3_5::cosize_v<SmemLayoutScale>;
    } 
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Type not handled in scale smem allocation.");
    }
  }

  // These methods use some the public members of the class. For that reason, we define them after the public section.
  static constexpr uint32_t
  compute_tma_transaction_bytes() {
    constexpr uint32_t a_bytes = cutlass_3_5::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(cute_3_5::sizeof_bits_v<InternalElementA>));
    constexpr uint32_t b_bytes = cutlass_3_5::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(cute_3_5::sizeof_bits_v<InternalElementB>));

    constexpr uint32_t baseline_bytes = a_bytes + b_bytes;

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return baseline_bytes;
    }
    else if constexpr (ModeHasScales) {
      constexpr uint32_t scale_tx_bytes = cutlass_3_5::bits_to_bytes(size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) * static_cast<uint32_t>(cute_3_5::sizeof_bits_v<ElementScale>));
      static_assert(scale_tx_bytes % 128 == 0, "Each scale stage must be 128B aligned."); // required by TMA
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return baseline_bytes + scale_tx_bytes;
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        // Scale and zero share smem layout
        constexpr uint32_t zero_tx_bytes = cutlass_3_5::bits_to_bytes(size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) * static_cast<uint32_t>(cute_3_5::sizeof_bits_v<ElementZero>));
        static_assert(zero_tx_bytes % 128 == 0, "Each zero stage must be 128B aligned."); // required by TMA
        return baseline_bytes + scale_tx_bytes + zero_tx_bytes;
      }
      else {
        static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Type not handled in tma transaction bytes computation.");
      }
    }
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Type not handled in tma transaction bytes computation.");
    }
  }

public:
  static constexpr size_t SmemAlignmentA = cutlass_3_5::detail::alignment_for_swizzle(SmemLayoutA{}); 

  static constexpr size_t SmemAlignmentB = cutlass_3_5::detail::alignment_for_swizzle(SmemLayoutB{});

  // Just pick the max alignment of A and B since it is required to be at least 128B
  static constexpr size_t SmemAlignmentScale = cute_3_5::max(SmemAlignmentA, SmemAlignmentB);

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");

  struct SharedStorage
  {
    static constexpr int scale_elements = elements_per_smem_scale();
    static constexpr int zero_elements = elements_per_smem_zero();
    struct TensorStorage : cute_3_5::aligned_struct<cute_3_5::max(SmemAlignmentA, SmemAlignmentB)> {
      cute_3_5::ArrayEngine<RealInternalElementA, cute_3_5::cosize_v<SmemLayoutA>> smem_A;
      cute_3_5::ArrayEngine<typename TiledMma::ValTypeB, cute_3_5::cosize_v<SmemLayoutB>> smem_B;
      cute_3_5::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
      cute_3_5::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A = nullptr;
    StrideA dA{};
    ElementB const* ptr_B = nullptr;
    StrideB dB{};
    ElementScale const* ptr_S = nullptr;
    NonVoidStrideScale dS{};
    int group_size = 0;
    ElementZero const* ptr_Z = nullptr;
    uint32_t mma_promotion_interval = 4;
  };

  // Device side kernel params
  struct Params {
  private:
    using Outer = CollectiveMma<DispatchPolicy, TileShape_, 
                                ElementAOptionalTuple, StrideA_, 
                                ElementBOptionalTuple, StrideB_,
                                TiledMma_, 
                                GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_,
                                TransformA_,
                                GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_,
                                TransformB_>;

  public:
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        make_tensor(Outer::get_logical_ptr(static_cast<InternalElementA const*>(nullptr)), repeat_like(InternalStrideA{}, int32_t(0)), InternalStrideA{}),
        SmemLayoutA{}(_,_,cute_3_5::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any

   using TMA_Scale = decltype(make_tma_copy(
        GmemTiledCopyScale{},
        make_tensor(Outer::get_logical_ptr(static_cast<NonVoidElementScale const*>(nullptr)), repeat_like(NonVoidStrideScale{}, int32_t(0)), NonVoidStrideScale{}),
        SmemLayoutScale{}(_,_,cute_3_5::Int<0>{}),
        ScaleTileShape{},
        _1{}));  // mcast along N mode for this M load, if any. Scale is ALWAYS loaded with A for RF kernel

   using TMA_Zero = decltype(make_tma_copy(
        GmemTiledCopyScale{},
        make_tensor(Outer::get_logical_ptr(static_cast<NonVoidElementZero const*>(nullptr)), repeat_like(NonVoidStrideScale{}, int32_t(0)), NonVoidStrideScale{}),
        SmemLayoutScale{}(_,_,cute_3_5::Int<0>{}),
        ScaleTileShape{},
        _1{}));  // mcast along N mode for this M load, if any. Scale is ALWAYS loaded with A for RF kernel

    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(Outer::get_logical_ptr(static_cast<InternalElementB const*>(nullptr)), repeat_like(InternalStrideB{}, int32_t(0)), InternalStrideB{}),
        SmemLayoutB{}(_,_,cute_3_5::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_Scale tma_load_scale;
    TMA_Zero tma_load_zero;
    int64_t scale_k;
    int group_size;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    if constexpr (SwapAB) {
      M = get<1>(problem_shape_MNKL);
      N = get<0>(problem_shape_MNKL);
    }

    InternalElementA const* ptr_A;
    InternalStrideA dA;
    InternalElementB const* ptr_B;
    InternalStrideB dB;

    if constexpr (not SwapAB) {
      ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
      ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);
      dA = args.dA;
      dB = args.dB;
    }
    else {
      ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_B);
      ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_A);
      dA = args.dB;
      dB = args.dA;
    }

    Tensor tensor_a = make_tensor(get_logical_ptr(ptr_A), make_layout(make_shape(M,K,L), dA));
    Tensor tensor_b = make_tensor(get_logical_ptr(ptr_B), make_layout(make_shape(N,K,L), dB));
    typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute_3_5::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any

    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute_3_5::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any

    typename Params::TMA_Scale tma_load_scale;
    typename Params::TMA_Zero tma_load_zero;
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return { tma_load_a, tma_load_b, tma_load_scale, tma_load_zero, 0, 0 };
    } 
    else if constexpr (ModeHasScales) {
      auto scale_k = (K + args.group_size - 1) / args.group_size;
      ElementScale const* ptr_S = args.ptr_S;
      StrideScale dS = args.dS;
      Tensor tensor_scale = make_tensor(get_logical_ptr(ptr_S), make_layout(make_shape(M,scale_k,L), dS));
      tma_load_scale = make_tma_copy(
          GmemTiledCopyScale{},
          tensor_scale,
          SmemLayoutScale{}(_,_,cute_3_5::Int<0>{}),
          ScaleTileShape{},
          _1{}); // mcast along N mode for this M load, if any

      if constexpr(KernelConversionMode == ConversionMode::ConvertAndScale) {
        return { tma_load_a, tma_load_b, tma_load_scale, tma_load_zero, scale_k, args.group_size };
      }
      else if constexpr(KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor tensor_zero = make_tensor(get_logical_ptr(args.ptr_Z), make_layout(make_shape(M,scale_k,L), dS));
        tma_load_zero = make_tma_copy(
            GmemTiledCopyScale{},
            tensor_zero,
            SmemLayoutScale{}(_,_,cute_3_5::Int<0>{}),
            ScaleTileShape{},
            _1{}); // mcast along N mode for this M load, if any
        return { tma_load_a, tma_load_b, tma_load_scale, tma_load_zero, scale_k, args.group_size };
      } else {
        static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");
      }
    } 
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in to_underlying_arguments.");
    }
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    
    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass_3_5::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass_3_5::detail::check_alignment<min_tma_aligned_elements_A>(cute_3_5::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass_3_5::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass_3_5::detail::check_alignment<min_tma_aligned_elements_B>(cute_3_5::make_shape(N,K,L), StrideB{});

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      implementable = implementable && (args.ptr_S == nullptr);
      implementable = implementable && (args.ptr_Z == nullptr);
    } 
    else if constexpr (ModeHasScales) {
      const int scale_mn = SwapAB ? N : M;
      const int scale_k = (K + args.group_size - 1) / args.group_size;
      constexpr int min_tma_aligned_elements_scale = tma_alignment_bits / cutlass_3_5::sizeof_bits<ElementScale>::value;
      implementable = implementable && cutlass_3_5::detail::check_alignment<min_tma_aligned_elements_scale>(cute_3_5::make_shape(scale_mn,scale_k,L), StrideScale{});
      implementable = implementable && (args.group_size == K || ((args.group_size % size<2>(TileShape{})) == 0));
      implementable = implementable && args.group_size != 0;
      implementable = implementable && (args.ptr_S != nullptr);

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        implementable = implementable && (args.ptr_Z == nullptr);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        constexpr int min_tma_aligned_elements_zero = tma_alignment_bits / cutlass_3_5::sizeof_bits<ElementZero>::value;
        implementable = implementable && cutlass_3_5::detail::check_alignment<min_tma_aligned_elements_zero>(cute_3_5::make_shape(scale_mn,scale_k,L), StrideScale{});
        implementable = implementable && (args.ptr_Z != nullptr);
      } 
      else {
        static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");
      }
    }
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in can_implement.");
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr uint32_t TmaTransactionBytes = compute_tma_transaction_bytes();

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute_3_5::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute_3_5::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // Nothing extra to do
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      cute_3_5::prefetch_tma_descriptor(mainloop_params.tma_load_scale.get_tma_descriptor());
    }
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      cute_3_5::prefetch_tma_descriptor(mainloop_params.tma_load_scale.get_tma_descriptor());
      cute_3_5::prefetch_tma_descriptor(mainloop_params.tma_load_zero.get_tma_descriptor());
    }  
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in TMA prefetch.");
    }
    
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});       // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});       // (BLK_N,BLK_K,n,k,l)

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute_3_5::make_tuple(gA_mkl, gB_nkl);
    } 
    else if constexpr (ModeHasScales) {
      auto scale_k = mainloop_params.scale_k;
      Tensor mS_mkl = mainloop_params.tma_load_scale.get_tma_tensor(make_shape(M,scale_k,L));      // (m,scale_k,l)
      Tensor gS_mkl = local_tile(mS_mkl, ScaleTileShape{}, make_coord(_,_));       // (BLK_M,BLK_Scale_K,m,scale_k,l)
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute_3_5::make_tuple(gA_mkl, gB_nkl, gS_mkl);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor mZ_mkl = mainloop_params.tma_load_zero.get_tma_tensor(make_shape(M,scale_k,L));      // (m,scale_k,l)
        Tensor gZ_mkl = local_tile(mZ_mkl, ScaleTileShape{}, make_coord(_,_));      // (BLK_M,BLK_Scale_K,m,scale_k,l)
        return cute_3_5::make_tuple(gA_mkl, gB_nkl, gS_mkl, gZ_mkl);
      }
      else {
        static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
      }
    } 
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in load_init.");
    }
  }  

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  /// This overload gets triggered when we have scales.
  template <
    class... Ts,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      cute_3_5::tuple<Ts...> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      static_assert(sizeof... (Ts) == 2, "Direct convert needs two inputs");
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      static_assert(sizeof... (Ts) == 3, "Scaled convert needs three inputs");
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      static_assert(sizeof... (Ts) == 4, "Scaled and zero convert needs four inputs");
    } 
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in TMA load.");
    }

    int lane_predicate = cute_3_5::elect_one_sync();

    if (lane_predicate) {
      Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
      Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)
      Tensor sA  = as_position_independent_swizzle_tensor(sA_);                                       // (BLK_M,BLK_K,PIPE)
      Tensor sB  = as_position_independent_swizzle_tensor(sB_);                                       // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A, B and Scales
      //
      
      constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                              // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                              // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;
      uint16_t mcast_mask_s = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute_3_5::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute_3_5::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }

      auto extra_input_partitions = partition_extra_tma_inputs(mainloop_params, load_inputs, shared_tensors, cluster_local_block_id, m_coord, l_coord);

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

        if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
          // Nothing extra to do.
        }
        else if constexpr (ModeHasScales) {
          auto tSgS = get<0>(extra_input_partitions);
          auto tSsS = get<1>(extra_input_partitions);

          // Temporary factor which will determine which k tile to reload from gmem. Needed so we don't modify tma transaction bytes
          // on the fly.
          // We must do a ceiling divide here to correctly handle with group_size == K. In that case, we don't require that K
          // is a multiple of the threadblock tile K
          const int ReloadFactor = (mainloop_params.group_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
          const int scale_load_k = *k_tile_iter / ReloadFactor; // This will always be 0 when group_size == K.
          copy(mainloop_params.tma_load_scale.with(*tma_barrier, mcast_mask_s), tSgS(_,_,_,scale_load_k), tSsS(_,_,_,write_stage));

          if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
            // Nothing extra to do
          } 
          else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
            auto tZgZ = get<2>(extra_input_partitions);
            auto tZsZ = get<3>(extra_input_partitions);
            copy(mainloop_params.tma_load_zero.with(*tma_barrier, mcast_mask_s), tZgZ(_,_,_,scale_load_k), tZsZ(_,_,_,write_stage));
          }
          else {
            static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for TMA copy op.");
          } 
        } 
        else {
          static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for TMA copy op.");
        }

        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute_3_5::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all 
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was 
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute_3_5::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute_3_5::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute_3_5::rank(InternalSmemLayoutAtomA{}) == 2, "InternalSmemLayoutAtomA must be rank 2.");
    static_assert(cute_3_5::rank(InternalSmemLayoutAtomB{}) == 2, "InternalSmemLayoutAtomB must be rank 2.");
    static_assert(!cute_3_5::is_void_v<InternalSmemCopyAtomA>,
      "SM90 GMMA mainloops must specify a non-void copy atom for RF sourced instructions.");
    static_assert(cute_3_5::is_void_v<InternalSmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // Obtain warp index
    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;
    
    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sA = as_position_independent_swizzle_tensor(sA_);                                      // (BLK_M,BLK_K,PIPE)
    
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = thread_mma.partition_A(sA);

    // Allocate fragments and descriptors
    Tensor tCrA_mma = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                      // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA_load = make_fragment_like<RealInternalElementA>(tCrA_mma);
    
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    //
    // Copy Atom A retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(InternalSmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);

    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA_load);                             // (CPY,CPY_M,CPY_K)

    // Compute the max vector length that can be used to copy A. This will match the vector width of the 
    // conversions used. It helps by allowing the compiler to convert using the same register that was used
    // to load the data from smem. This significantly reduces the need to move data among registers.
    // Note that this is correct even if copy fails to vectorize, since the granularity at which we perform
    // the conversion does not impact correctness.
    using A_CPY_VEC = decltype(max_common_vector(tCsA, tCrA_copy_view));

    // Partition of thread -> shared and thread -> RF
    auto partitioned_extra_info = partition_extra_mma_info(thread_mma, shared_tensors);
    auto copy_partitions_extra_info = retile_extra_mma_info(tiled_mma, partitioned_extra_info, warp_group_thread_idx);

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                                            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                                            // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));                                                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(accum);

    constexpr int K_BLOCK_MAX = size<2>(tCrA_load);
    
    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};
    // first k tile
    {
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();

      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

      // copy smem->rmem for A operand
      copy_A_and_extra_info(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
        partitioned_extra_info, copy_partitions_extra_info, 0, read_stage);

      transform_A_kblock(tCrA_load, A_CPY_VEC{}, tCrA_mma, partitioned_extra_info, 0);
      
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        if (k_block < K_BLOCK_MAX - 1) {
          copy_A_and_extra_info(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
            partitioned_extra_info, copy_partitions_extra_info, k_block + 1, read_stage);
          transform_A_kblock(tCrA_load, A_CPY_VEC{}, tCrA_mma, partitioned_extra_info, k_block + 1);
        }
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute_3_5::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();
      }     

      --k_tile_count;
      if (k_tile_count > 0) {
        // Wait for K_BLOCK_MAX - 1 to be in flight to ensure that it is safe to overwrite the A registers for the first mma.
        warpgroup_wait<K_BLOCK_MAX - 1>(); 
        pipeline.consumer_wait(smem_pipe_read, barrier_token);
        copy_A_and_extra_info(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
          partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read.index());
        transform_A_kblock(tCrA_load, A_CPY_VEC{}, tCrA_mma, partitioned_extra_info, 0);
      }
    }

    if (k_tile_count == 0) {
      return;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;

      warpgroup_fence_operand(accum);
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute_3_5::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();

        warpgroup_wait<K_BLOCK_MAX - 1>();
        if (k_block == K_BLOCK_MAX - 1) {
          // We have K_BLOCK_MAX - 1 GMMA instructions pending for this stage, so we can release prior barrier
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }

        if (k_block == 0) {
          barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        }

        if (k_block == K_BLOCK_MAX - 1) { 
          pipeline.consumer_wait(smem_pipe_read, barrier_token);
          copy_A_and_extra_info(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
            partitioned_extra_info, copy_partitions_extra_info, 0, smem_pipe_read.index());
          transform_A_kblock(tCrA_load, A_CPY_VEC{}, tCrA_mma, partitioned_extra_info, 0);
        } 
        else {
          copy_A_and_extra_info(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
            partitioned_extra_info, copy_partitions_extra_info, k_block + 1, read_stage);
          transform_A_kblock(tCrA_load, A_CPY_VEC{}, tCrA_mma, partitioned_extra_info, k_block + 1);
        }
      }
      warpgroup_fence_operand(accum);

    }

    warpgroup_fence_operand(accum);

    {
      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      warpgroup_fence_operand(accum);
      
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {

        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute_3_5::gemm(tiled_mma, tCrA_mma(_,_,k_block), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();
        warpgroup_wait<K_BLOCK_MAX - 1>();
        if (k_block == K_BLOCK_MAX - 1) {
          // release prior barrier
          pipeline.consumer_release(smem_pipe_release);             // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }

        if (k_block < K_BLOCK_MAX - 1) {
          copy_A_and_extra_info(smem_tiled_copy_A, tCsA, tCrA_copy_view, 
            partitioned_extra_info, copy_partitions_extra_info, k_block + 1, read_stage);
          transform_A_kblock(tCrA_load, A_CPY_VEC{}, tCrA_mma, partitioned_extra_info, k_block + 1);
        }
      }
    }

    warpgroup_fence_operand(accum);
  }
  
  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = 1;
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);
    
    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

private:
  /// Utilities for any additional inputs inside of the TMA load
  template <class... Ts>
  CUTLASS_DEVICE
  auto partition_extra_tma_inputs(
    Params const& mainloop_params,
    cute_3_5::tuple<Ts...> const& load_inputs,
    TensorStorage& shared_tensors,
    uint2 const& cluster_local_block_id,
    int const m_coord, 
    int const l_coord) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute_3_5::tuple{};
    } 
    else if constexpr (ModeHasScales) {
      Tensor sS  = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{}); // (BLK_M,BLK_K,PIPE)
      Tensor gS_mkl = get<2>(load_inputs);
      auto block_tma_s = mainloop_params.tma_load_scale.get_slice(cluster_local_block_id.y);
      Tensor gS = gS_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)

      Tensor tSgS = block_tma_s.partition_S(gS);                                                     // (TMA,TMA_M,TMA_K,k)
      Tensor tSsS = block_tma_s.partition_D(sS);                                                     // (TMA,TMA_M,TMA_K,PIPE)
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute_3_5::make_tuple(tSgS, tSsS);
      } 
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ  = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()), SmemLayoutScale{}); // (BLK_M,BLK_K,PIPE)
        Tensor gZ_mkl = get<3>(load_inputs);
        auto block_tma_z = mainloop_params.tma_load_zero.get_slice(cluster_local_block_id.y);
        Tensor gZ = gZ_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)

        Tensor tZgZ = block_tma_z.partition_S(gZ);                                                     // (TMA,TMA_M,TMA_K,k)
        Tensor tZsZ = block_tma_z.partition_D(sZ);                                                     // (TMA,TMA_M,TMA_K,PIPE)
        return cute_3_5::make_tuple(tSgS, tSsS, tZgZ, tZsZ);          
      }
      else {
        static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");      
      }
    }
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled for input partitioning.");      
    }
  }

  /// Utilities for partitioning extra inputs for loading from smem in the mainloop.
  template <class ThreadMma>
  CUTLASS_DEVICE 
  auto partition_extra_mma_info(
    ThreadMma const& thread_mma,
    TensorStorage& shared_tensors) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // noting to do
      return cute_3_5::tuple{};
    }
    else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutScale{});    // (BLK_M,BLK_SCALE_K,PIPE)
      Tensor tCsS = thread_mma.partition_A(sS);
      Tensor tCrS = make_tensor<ElementScale>(thread_mma.partition_fragment_A(sS(_,_,Int<0>{})).shape()); 

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute_3_5::make_tuple(tCsS, tCrS);
      }
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()), SmemLayoutScale{});    // (BLK_M,BLK_SCALE_K,PIPE)
        Tensor tCsZ = thread_mma.partition_A(sZ);
        Tensor tCrZ = make_tensor<ElementZero>(thread_mma.partition_fragment_A(sZ(_,_,Int<0>{})).shape()); 
        return cute_3_5::make_tuple(tCsS, tCrS, tCsZ, tCrZ);
      }
      else {
        static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      }
    } 
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
    }
  }

  /// Returns the tiled copy and copy views for the extra inputs.
  template <class TiledMma, class... Ts>
  CUTLASS_DEVICE
  auto retile_extra_mma_info(
    TiledMma const& tiled_mma,
    cute_3_5::tuple<Ts...>& partitioned_extra_info,
    int const warp_group_thread_idx) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // noting to do
      return cute_3_5::tuple{};
    }
    else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S = make_tiled_copy_A(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S   = smem_tiled_copy_S.get_thread_slice(warp_group_thread_idx);
      Tensor tCrS_copy_view  = smem_thr_copy_S.retile_D(cute_3_5::get<1>(partitioned_extra_info));        // (CPY,CPY_M,CPY_K)
      
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute_3_5::make_tuple(smem_tiled_copy_S, tCrS_copy_view);
      } 
      else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        Tensor tCrZ_copy_view  = smem_thr_copy_S.retile_D(cute_3_5::get<3>(partitioned_extra_info));      // (CPY,CPY_M,CPY_K)
        return cute_3_5::make_tuple(smem_tiled_copy_S, tCrS_copy_view, tCrZ_copy_view);
      } 
      else {
        static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      }
    } 
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
    }
  }

  /// Utilities to copy A and extra inputs from smem to RF
  template <class SmemTiledCopyA,
            class TensorASmemView,
            class TensorACopyView,
            class... Ts,
            class... Us
            >
  CUTLASS_DEVICE
  void copy_A_and_extra_info(
    SmemTiledCopyA const& smem_tiled_copy_A,
    TensorASmemView const& tCsA,
    TensorACopyView& tCrA_copy_view,
    cute_3_5::tuple<Ts...> const& partitioned_mma_extra_info,
    cute_3_5::tuple<Us...> const& tiled_copy_and_views,
    int k_block,
    int read_stage) {

    copy(smem_tiled_copy_A, tCsA(_,_,k_block,read_stage), tCrA_copy_view(_,_,k_block));

    if (k_block == 0) {
      // We are starting a new k-tile so copy the scale
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // nothing to do
      } 
      else if constexpr (ModeHasScales) {
        auto smem_tiled_copy_S = cute_3_5::get<0>(tiled_copy_and_views);
        auto tCrS_copy_view    = cute_3_5::get<1>(tiled_copy_and_views);
        auto tCsS              = cute_3_5::get<0>(partitioned_mma_extra_info);
        copy(smem_tiled_copy_S, tCsS(_,_,k_block,read_stage), tCrS_copy_view(_,_,k_block));
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
          // Nothing extra to do
        } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
          auto tCsZ              = cute_3_5::get<2>(partitioned_mma_extra_info);
          auto tCrZ_copy_view    = cute_3_5::get<2>(tiled_copy_and_views);
          copy(smem_tiled_copy_S, tCsZ(_,_,k_block,read_stage), tCrZ_copy_view(_,_,k_block));
        } else {
          static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");         
        }
      } 
      else {
        static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "Conversion mode not handled in A -> RF path.");
      }
    }
  }

  /// Utilities to transform A.
  template <class TCrA_load,
            int VectorWidthA, 
            class TCrA_mma,
            class... Ts>
  CUTLASS_DEVICE
  void transform_A_kblock(
    TCrA_load const& tCrA_load, 
    cute_3_5::Int<VectorWidthA> vec_A,
    TCrA_mma& tCrA_mma,
    cute_3_5::tuple<Ts...> const& partitioned_extra_info,
    int const k_block) {

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      transform_internal_A(tCrA_load(_, _, k_block), vec_A, tCrA_mma(_, _, k_block));
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      auto tCrS = cute_3_5::get<1>(partitioned_extra_info);
      transform_internal_A(tCrA_load(_, _, k_block), vec_A, make_fragment_like<ElementScale>(tCrA_mma)(_, _, k_block), tCrS(_, _, 0), tCrA_mma(_, _, k_block));
    } 
    else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
      auto tCrS = cute_3_5::get<1>(partitioned_extra_info);
      auto tCrZ = cute_3_5::get<3>(partitioned_extra_info);
      transform_internal_A(tCrA_load(_, _, k_block), 
                           vec_A, 
                           make_fragment_like<ElementScale>(tCrA_mma)(_, _, k_block), 
                           tCrS(_, _, 0),
                           tCrZ(_, _, 0),
                           make_fragment_like<ElementScale>(tCrZ)(_, _, 0), 
                           tCrA_mma(_, _, k_block));        
    }
    else {
      static_assert(cutlass_3_5::detail::dependent_false<KernelSchedule>, "No A data is loaded.");
    }
  }

  /// Utilities for transforming the A operand prior to issuing tensorcore math.
  template <class EngineIn, 
            class EngineOut, 
            class TensorLayout,
            int ConversionVectorWidth = cosize_v<TensorLayout>>
  CUTLASS_DEVICE void
  convert_tensor(
    Tensor<EngineIn,TensorLayout> const& in, 
    Tensor<EngineOut,TensorLayout>& out, 
    cute_3_5::Int<ConversionVectorWidth> width = {}) {

    /// This is an element-wise conversion where we expect both tensors to have the same layout.
    /// As a result, we can cast as a cutlass_3_5 array to use the fast numeric converters without 
    /// worrying about indexing into the layout.
    constexpr int N = cosize_v<TensorLayout>; 

    /// The inputs must be backed by registers & be statically sized.
    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for A conversion must come from registers");
    static_assert(is_static_v<TensorLayout>, "Tensor layout for the conversion must be static");
    static_assert(cosize_v<TensorLayout> == size(TensorLayout{}), "Cosize and size of the layout must be equal.");
    static_assert(N % ConversionVectorWidth == 0, "Conversion vector width must divide cosize of the tensor layout.");

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;
  
    using SrcArray = cutlass_3_5::Array<SrcType, ConversionVectorWidth>;
    using DstArray = cutlass_3_5::Array<DstType, ConversionVectorWidth>;

    constexpr cutlass_3_5::FloatRoundStyle RoundStyle = cutlass_3_5::FloatRoundStyle::round_to_nearest;
    using Converter = cutlass_3_5::NumericArrayConverter<DstType, SrcType, ConversionVectorWidth, RoundStyle>;

    constexpr int NumIterations = N / ConversionVectorWidth;

    for (int ii = 0; ii < NumIterations; ++ii) {
      SrcArray const* src_array_ptr = reinterpret_cast<SrcArray const*>(raw_pointer_cast(in.data())) + ii;
      DstArray* dst_array_ptr = reinterpret_cast<DstArray*>(raw_pointer_cast(out.data())) + ii;
      *dst_array_ptr = Converter::convert(*src_array_ptr);
    }
  }

  template <class EngineIn, 
            class EngineOut, 
            class TensorLayout,
            int A_VectorConversionWidth>
  CUTLASS_DEVICE void
  transform_internal_A(
    Tensor<EngineIn,TensorLayout>&& in,
    cute_3_5::Int<A_VectorConversionWidth> a_vec_width,
    Tensor<EngineOut,TensorLayout>&& out) {

    convert_tensor(in, out, a_vec_width);
  }

  template <class EngineIn, 
            class EngineInputBuffer, 
            class EngineScale, 
            class EngineOut, 
            class TensorLayout,
            int A_VectorConversionWidth>
  CUTLASS_DEVICE void
  transform_internal_A(
    Tensor<EngineIn,TensorLayout>&& in,
    cute_3_5::Int<A_VectorConversionWidth> a_vec_width,
    Tensor<EngineInputBuffer,TensorLayout>&& converted_inputs,
    Tensor<EngineScale,TensorLayout>&& scales,
    Tensor<EngineOut,TensorLayout>&& out) {

    static_assert(cute_3_5::is_same_v<typename EngineInputBuffer::value_type, typename EngineScale::value_type>,  
      "Type of the engine input buffer must equal the scale buffer");
    
    // First, we upcast the inputs to the scale type
    convert_tensor(in, converted_inputs, a_vec_width);

    // Apply scales and broadcast across inputs, store in converted_inputs
    cute_3_5::transform(converted_inputs, scales, converted_inputs, cute_3_5::multiplies{});

    // Finally, we convert the scaled inputs to the mma type.
    convert_tensor(converted_inputs, out);
  }

  template <class EngineIn, 
            class EngineInputBuffer, 
            class EngineScale,
            class EngineZero,
            class EngineZeroBuffer,
            class EngineOut, 
            class TensorLayout,
            int A_VectorConversionWidth>
  CUTLASS_DEVICE void
  transform_internal_A(
    Tensor<EngineIn,TensorLayout>&& in,
    cute_3_5::Int<A_VectorConversionWidth> a_vec_width,
    Tensor<EngineInputBuffer,TensorLayout>&& converted_inputs,
    Tensor<EngineScale,TensorLayout>&& scales,
    Tensor<EngineZero,TensorLayout>&& zeros,
    Tensor<EngineZeroBuffer,TensorLayout>&& converted_zeros,
    Tensor<EngineOut,TensorLayout>&& out) {

    static_assert(cute_3_5::is_same_v<typename EngineInputBuffer::value_type, typename EngineScale::value_type>,  
      "Type of the engine input buffer must equal the scale buffer");

    static_assert(cute_3_5::is_same_v<typename EngineZeroBuffer::value_type, typename EngineScale::value_type>,  
      "Type of the engine zero buffer must equal the scale buffer");
    
    // First, we upcast the inputs to the scale type
    convert_tensor(in, converted_inputs, a_vec_width);
    convert_tensor(zeros, converted_zeros);

    // Apply scales and broadcast across inputs, store in converted_inputs
    cute_3_5::transform(converted_inputs, scales, converted_inputs, cute_3_5::multiplies{});
    cute_3_5::transform(converted_inputs, converted_zeros, converted_inputs, cute_3_5::plus{});

    // Finally, we convert the scaled inputs to the mma type.
    convert_tensor(converted_inputs, out);
  } 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_3_5::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
