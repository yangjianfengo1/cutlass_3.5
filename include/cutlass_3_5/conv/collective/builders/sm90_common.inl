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

#include "cutlass_3_5/layout/tensor.h"
#include "cutlass_3_5/arch/mma.h"
#include "cutlass_3_5/conv/convolution.h"
#include "cutlass_3_5/conv/dispatch_policy.hpp"
#include "cutlass_3_5/detail/layout.hpp"
#include "cutlass_3_5/gemm/collective/builders/sm90_common.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_3_5::conv::collective::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Maps a rank-1 cute_3_5::Shape<> representing the cluster shape on to the IM2COL TMA atom that should be used with it
template <class UnimodalClusterShape>
constexpr auto
sm90_cluster_shape_to_im2col_tma_atom(UnimodalClusterShape unimodal_cluster_shape) {
  static_assert(cute_3_5::rank(unimodal_cluster_shape) == 1,
    "Use this function to figure out TMA for each mode individually.");

  if constexpr (cute_3_5::size(unimodal_cluster_shape) == 1) {
    return cute_3_5::SM90_TMA_LOAD_IM2COL{};
  }
  else {
    return cute_3_5::SM90_TMA_LOAD_IM2COL_MULTICAST{};
  }
}

// Collective tile traits struct that serves as a type list containing a tensor's mem layouts and atoms for the
template<
  class GmemTiledCopy_,
  class SmemLayout_,
  class SmemCopyAtom_ = void
>
struct Sm90ImplicitGemmTileTraits {
  using GmemTiledCopy = GmemTiledCopy_;
  using SmemLayout = SmemLayout_;
  using SmemCopyAtom = SmemCopyAtom_;
};

// Accepts a cutlass_3_5::layout::Tensor tag and computes the corresponding spatial dimension count
template <class GmemLayoutTagA, class GmemLayoutTagB>
constexpr int
gmem_layout_tags_to_spatial_dims() {
  static_assert(cute_3_5::is_same_v<GmemLayoutTagA, GmemLayoutTagB>);
  if constexpr      (cute_3_5::is_same_v<GmemLayoutTagA, cutlass_3_5::layout::TensorNWC>) {
    return 1;
  }
  else if constexpr (cute_3_5::is_same_v<GmemLayoutTagA, cutlass_3_5::layout::TensorNHWC>) {
    return 2;
  }
  else if constexpr (cute_3_5::is_same_v<GmemLayoutTagA, cutlass_3_5::layout::TensorNDHWC>) {
    return 3;
  }
  else {
    static_assert(cutlass_3_5::detail::dependent_false<GmemLayoutTagA>);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_3_5::conv::collective::detail

/////////////////////////////////////////////////////////////////////////////////////////////////
