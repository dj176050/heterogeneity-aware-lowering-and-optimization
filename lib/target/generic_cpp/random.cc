//===- random.cc ----------------------------------------------------------===//
//
// Copyright (C) 2019-2021 Alibaba Group Holding Limited.
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

#include "halo/lib/ir/creator_instructions.h"
#include "halo/lib/target/generic_cxx/generic_cxx_codegen.h"

namespace halo {

void GenericCXXCodeGen::RunOnInstruction(RandomUniformInst* inst) {
  const auto& ret_type = inst->GetResultType();

  CXXValue ret(inst->GetName(), TensorTypeToCXXType(ret_type, false));
  EmitODLACall(ret, "odla_Fill", ret_type, "ODLA_RandomUniform",
               inst->GetMinval(), inst->GetMaxval(), inst->GetSeed());
  ir_mapping_[*inst] = ret;
}

} // namespace halo
