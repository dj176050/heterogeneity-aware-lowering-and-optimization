//===- inceptionv2_test.cc ------------------------------------------------===//
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
// RUN: %halo_compiler --disable-broadcasting -enable-bf16 -target cxx %src_dir/tests/parser/onnx/inception_v2.onnx -o %t.cc
// RUN: %cxx -O3 -c -o %t.o %t.cc -I%odla_path/include
// RUN: %cxx -O3 -g -DBATCH=4 -DTIMING_TEST -DITER_TEST=10 %include %s %t_nc.o %t_nc.bin %odla_link -lodla_dnnl -o %t_nc.exe
// RUN: %t.exe | FileCheck %s

// CHECK: Result verified
// clang-format on

#define TEST_SET 0

#include "inceptionv2_data.h"
#include "test_util.h"

extern "C" {
void inception_v2(const float* in, float* out);
}

#ifndef COMPARE_ERROR
#define COMPARE_ERROR 1e-3
#endif

int main(int argc, char** argv) {
  float out[1000];
  inception_v2(test_input, out);
  if (Verify(out, test_output_ref, sizeof(out) / sizeof(out[0]),
             COMPARE_ERROR)) {
    std::cout << "Result verified\n";
#ifdef TIMING_TEST
    auto begin_time = Now();
    inception_v2(test_input, out);
    auto end_time = Now();
    std::cout << "Elapse time: " << GetDuration(begin_time, end_time)
              << " seconds\n";
#endif
    return 0;
  }
  std::cout << " Failed\n";
  return 1;
}
