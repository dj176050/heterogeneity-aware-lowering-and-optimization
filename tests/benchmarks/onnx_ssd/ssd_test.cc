//===- ssd_test.cc --------------------------------------------------------===//
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

// clang-format off
// Testing CXX Code Gen using Open DL API on DNNL
// RUN: %halo_compiler --disable-broadcasting -enable-bf16 -target cxx %src_dir/tests/parser/onnx/ssd.onnx -o %t.cc
// RUN: %cxx -O3 -c -o %t.o %t.cc -I%odla_path/include
// RUN: %cxx -O3 -g -DBATCH=4 -DTIMING_TEST -DITER_TEST=10 %include %s %t_nc.o %t_nc.bin %odla_link -lodla_dnnl -o %t_nc.exe
// RUN: %t.exe | FileCheck %s

// CHECK: Result verified
// clang-format on
#include <algorithm>

#include "ssd_data.h"
#include "test_util.h"

extern "C" {
void ssd(const float* in, float* bboxs, int64_t* labels, float* scores);
}

#ifndef COMPARE_ERROR
#define COMPARE_ERROR 1e-3
#endif

#ifndef BOX_ERROR
#define BOX_ERROR 200
#endif

int main(int argc, char** argv) {
  // declare as static to prevent stack overflow
  static float bboxs[1 * nbox * 4];
  static int64_t labels[1 * nbox];
  static float scores[1 * nbox];
  int num_detections;

  ssd(test_input, bboxs, labels, scores);
#ifdef DEBUG_TEST
  std::cout << "Detected " << nbox << "boxes: \n";
  for (int i = 0; i < nbox; ++i) {
    std::cout << bboxs[i * 4] << ", " << bboxs[i * 4 + 1] << ", "
              << bboxs[i * 4 + 2] << ", " << bboxs[i * 4 + 3] << ", ";
    std::cout << labels[i] << ", " << scores[i] << "\n";
  }
#endif
  bool flag0 =
      Verify(bboxs, test_output_ref0,
             std::min((size_t)BOX_ERROR, sizeof(bboxs) / sizeof(bboxs[0])),
             COMPARE_ERROR);
  bool flag1 =
      Verify(labels, test_output_ref1,
             std::min((size_t)BOX_ERROR, sizeof(labels) / sizeof(labels[0])),
             COMPARE_ERROR);
  bool flag2 =
      Verify(scores, test_output_ref2,
             std::min((size_t)BOX_ERROR, sizeof(scores) / sizeof(scores[0])),
             COMPARE_ERROR);

  if (flag0 && flag1 && flag2) {
    std::cout << "Result verified\n";
#ifdef TIMING_TEST
    auto begin_time = Now();
    ssd(test_input, bboxs, labels, scores);
    auto end_time = Now();
    std::cout << "Elapse time: " << GetDuration(begin_time, end_time)
              << " seconds\n";
#endif
    return 0;
  }
  std::cout << " Failed\n";
  return 1;
}
