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

#include <vector_types.h>
#include <cutlass_3_5/numeric_types.h>
#include <cutlass_3_5/numeric_size.h>

#include <cute_3_5/numeric/int.hpp>
#include <cute_3_5/numeric/real.hpp>

namespace cute_3_5 {

template <typename T>
struct sizeof_bits : public cutlass_3_5::sizeof_bits<T> {};

// DO NOT change auto to int, sizeof_bits<sparse_elem> use integral_ratio instead of int 
template <class T>
static constexpr auto sizeof_bits_v = sizeof_bits<T>::value;

using cutlass_3_5::bits_to_bytes;

using cutlass_3_5::is_subbyte;

template <class T>
static constexpr auto is_subbyte_v = is_subbyte<T>::value;

using cutlass_3_5::half_t;
using cutlass_3_5::bfloat16_t;

using cutlass_3_5::tfloat32_t;

// Umbrella floating-point 8-bit data type : type_erased_dynamic_float8_t
// This umbrella datatype can be enabled when a user provides a specific
// datatype in runtime argument list.
using cutlass_3_5::type_erased_dynamic_float8_t;
using cutlass_3_5::float_e4m3_t;
using cutlass_3_5::float_e5m2_t;

using cutlass_3_5::uint1b_t;
using cutlass_3_5::int2b_t;
using cutlass_3_5::uint2b_t;
using cutlass_3_5::int4b_t;
using cutlass_3_5::uint4b_t;
using cutlass_3_5::bin1_t;

} // end namespace cute_3_5
