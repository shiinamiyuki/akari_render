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
#include <akari/util.h>
#include <chrono>
namespace akari {
    class Timer {
        using TP = decltype(std::chrono::high_resolution_clock::now());
        TP t0, t1;

      public:
        void start() { t0 = std::chrono::high_resolution_clock::now(); }
        void stop() { t1 = std::chrono::high_resolution_clock::now(); }
        double elapsed_seconds() const {
            std::chrono::duration<double> diff = t1 - t0;
            return diff.count();
        }
    };
} // namespace akari