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

/*! \file
    \brief Utilities to convert a CuTe tuple to a GemmCoord or BatchedGemmCoord
*/

#pragma once

#include "cute_3_5/layout.hpp"
#include "cutlass_3_5/gemm_coord.h"

namespace cutlass_3_5 {
namespace gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Tuple>
CUTLASS_HOST_DEVICE
auto
to_gemm_coord(Tuple tuple) {
  static_assert(cute_3_5::rank(tuple) <= 4, "Can only convert tuples of rank <= 4.");

  if constexpr (cute_3_5::rank(tuple) <= 3) {
    auto tuple_mnk = cute_3_5::append<3>(tuple, cute_3_5::Int<0>{});
    return GemmCoord(cute_3_5::size<0>(tuple_mnk), cute_3_5::size<1>(tuple_mnk), cute_3_5::size<2>(tuple_mnk));
  }
  else {
    return BatchedGemmCoord(cute_3_5::size<0>(tuple), cute_3_5::size<1>(tuple), cute_3_5::size<2>(tuple), cute_3_5::size<3>(tuple));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace cutlass_3_5

/////////////////////////////////////////////////////////////////////////////////////////////////
