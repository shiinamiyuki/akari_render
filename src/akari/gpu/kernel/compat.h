// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#if defined(AKR_BACKEND_CUDA) || defined(__CUDACC__)
#    if defined(AKR_BACKEND_CUDA)
#        include <cuda.h>

#    endif
#    define knl_device __device__
#    define knl_inline inline
#    define knl_device_inline knl_inline knl_device
#    define ms_global
#    define ms_local
#endif