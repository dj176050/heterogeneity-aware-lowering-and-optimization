# ==============================================================================
# Copyright (C) 2019-2020 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
# ==============================================================================

set(SRCS
  common/cast.cc
  common/gather.cc
  common/onehot.cc
  common/pad.cc
  common/reduce.cc
  common/slice.cc
  math/add.cc
  math/div.cc
  math/erf.cc
  math/floor.cc
  math/matmul.cc
  math/mul.cc
  math/sqrt.cc
  math/sub.cc
  math/transpose.cc
  nn/batchnorm.cc
  nn/conv.cc
  nn/pooling.cc
  nn/relu.cc
  nn/softmax.cc
)
add_library(RT_GENERIC ${SRCS})

set(OPT_FLAGS -O3)
target_compile_options(RT_GENERIC PRIVATE -emit-llvm ${OPT_FLAGS} -fno-exceptions -fno-unwind-tables)