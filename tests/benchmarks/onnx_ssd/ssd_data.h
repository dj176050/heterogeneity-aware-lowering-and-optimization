//===- ssd_data.h ---------------------------------------------------------===//
//
// Copyright (C) 2019-2020 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TESTS_BENCHMARKS_ONNX_SSD_DATA_H
#define TESTS_BENCHMARKS_ONNX_SSD_DATA_H

#include <cstdint>

static const int nbox = 200;

static const float test_input[1 * 3 * 1200 * 1200] = {
#include "input_data_0.txt"
};

static const float test_output_ref0[1 * nbox * 4] = {
#include "bboxes.txt"
};

static const int64_t test_output_ref1[1 * nbox] = {
#include "labels.txt"
};

static const float test_output_ref2[1 * nbox] = {
#include "scores.txt"
};

#endif // TESTS_BENCHMARKS_ONNX_SSD_DATA_H
