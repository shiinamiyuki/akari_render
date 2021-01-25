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

    static constexpr int BLUENOISE_RESOLUTION = 128;
    static constexpr int NUM_BLUENOISE_TEXTURES = 48;

    extern const uint16_t BlueNoiseTextures[NUM_BLUENOISE_TEXTURES][BLUENOISE_RESOLUTION][BLUENOISE_RESOLUTION];

    // Blue noise lookup functions
    inline float blue_nosie(int textureIndex, ivec2 p) {
        AKR_CHECK(textureIndex >= 0 && p.x >= 0 && p.y >= 0);
        textureIndex %= NUM_BLUENOISE_TEXTURES;
        int x = p.x % BLUENOISE_RESOLUTION, y = p.y % BLUENOISE_RESOLUTION;
        return BlueNoiseTextures[textureIndex][x][y] / 65535.f;
    }

} // namespace akari
