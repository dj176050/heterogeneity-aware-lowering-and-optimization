#include "resnet_test.in"

// clang-format off
// Testing CXX Code Gen using NCHW on Open DL API on DNNL (MKLDNN)
// RUN: %halo_compiler -target cxx %src_dir/tests/parser/tensorflow/resnet50_v2.pb -o %t_nc.cc -enable-bf16 -fuse-conv-bias -batch-size=4 -disable-broadcasting -reorder-data-layout=channel-first
// RUN: %cxx -O3 -g -I%odla_path/include -c -o %t_nc.o %t_nc.cc
// RUN: %cxx -O3 -g -DBATCH=4 -DTIMING_TEST -DITER_TEST=10 %include %s %t_nc.o %t_nc.bin %odla_link -lodla_dnnl -o %t_nc.exe
// RUN: %t_nc.exe | FileCheck %s
// CHECK: Result verified