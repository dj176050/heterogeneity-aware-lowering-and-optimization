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

add_odla_library(odla_popart SHARED common.cc odla_compute.cc
                               odla_ops_math.cc odla_ops_nn.cc
                               odla_ops_process.cc odla_ops.cc)
find_library(popart NAMES popart PATHS ${POPART_ROOT} PATH_SUFFIXES lib
             NO_DEFAULT_PATH)
find_library(custom_ops NAMES custom_ops PATHS ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops
             PATH_SUFFIXES lib NO_DEFAULT_PATH)
find_package(popart CONFIG REQUIRED)
set_property(TARGET odla_popart PROPERTY CXX_STANDARD 14)

target_compile_definitions(odla_popart PRIVATE ONNX_NAMESPACE=onnx)
target_link_libraries(odla_popart ODLA ${custom_ops} popart-only)
add_custom_command(TARGET odla_popart
	  POST_BUILD
	  COMMAND ${CMAKE_COMMAND} -E
	  copy ${custom_ops} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
	  )
