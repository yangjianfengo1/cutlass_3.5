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
    \brief Defines layout functions used by TensorRef and derived classes. 

    Layout functions map logical coordinates to linear memory. They often require additional
    data to describe strides between elements.

    Layout functions must implement all members in the public interface of IdentityTensorLayout<>
    defined in cutlass_3_5/tensor_ref.h.
*/
#pragma once

#include "cutlass_3_5/cutlass.h"
#include "cutlass_3_5/matrix_coord.h"
#include "cutlass_3_5/layout/matrix.h"
#include "cutlass_3_5/layout/pitch_linear.h"
#include "cutlass_3_5/layout/tensor.h"
#include "cutlass_3_5/layout/vector.h"

#include "cutlass_3_5/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass_3_5/layout/tensor_op_multiplicand_sm75.h"
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_3_5 {
namespace layout {

///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace layout
} // namespace cutlass_3_5

///////////////////////////////////////////////////////////////////////////////////////////////////
