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
#include <akari/render.h>
namespace akari::render::mlt {
    struct RadianceRecord {
        ivec2 p_film;
        Spectrum radiance;
    };
    struct MarkovChain {
        explicit MarkovChain(MLTSampler sampler):sampler(sampler){}
        Sampler sampler;
        RadianceRecord current;
    };
} // namespace akari::render::mlt