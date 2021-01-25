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

// Adopted from pbrt-v4
#pragma once

#include <akari/util.h>

namespace akari {

    // PMJ02BN Table Declarations
    constexpr int N_PMJ02BN_SETS = 5;
    constexpr int N_PMJ02BN_SAMPLES = 65536;
    extern const uint32_t pmj02bnSamples[N_PMJ02BN_SETS][N_PMJ02BN_SAMPLES][2];

    // PMJ02BN Table Inline Functions
    inline vec2 pmj02bn(int setIndex, int sampleIndex) {
        setIndex %= N_PMJ02BN_SETS;
        AKR_CHECK(sampleIndex < N_PMJ02BN_SAMPLES);
        sampleIndex %= N_PMJ02BN_SAMPLES;

        // Double precision is key here for the pixel sample sorting, but not
        // necessary otherwise.
        return vec2(pmj02bnSamples[setIndex][sampleIndex][0] * 0x1p-32,
                    pmj02bnSamples[setIndex][sampleIndex][1] * 0x1p-32);
    }

} // namespace akari
