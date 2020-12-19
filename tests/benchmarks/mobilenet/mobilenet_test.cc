//===- mobilenet_test.cc --------------------------------------------------===//
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

// Testing CXX Code Gen using Open DL API on TensorRT
// RUN: %halo_compiler -enable-bf16 -target cxx %src_dir/tests/parser/onnx/mobilenet_v2.onnx -o %t.cc
// RUN: %cxx -c -O3 -o %t.o %t.cc -I%odla_path/include
// RUN: %cxx -c -O3 -DTIMING_TEST -DITER_TEST=1000 -DCOMPARE_ERROR=1e-4 %include %s -o %t_main.o
// RUN: %cxx %t_main.o %t.o %t.bin %odla_link -lodla_dnnl -o %t.exe
// RUN: %t.exe | FileCheck %s

// CHECK: Result verified

#include "mobilenet_test.in"
