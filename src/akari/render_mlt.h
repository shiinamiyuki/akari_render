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
        ivec2 p_film = ivec2(0);
        Spectrum radiance = Spectrum(0);
    };
    struct MarkovChain {
        explicit MarkovChain(MLTSampler sampler) : sampler(sampler) {}
        Sampler<CPU>  sampler;
        RadianceRecord current;
    };
    inline auto T(const Spectrum &s) { return hmax(s); };
    struct MLTStats {
        AtomicDouble acc_b;
        std::atomic_uint64_t n_large;
        std::atomic_uint64_t accepts, rejects;
        MLTStats() : acc_b(0.0), n_large(0), accepts(0), rejects(0) {}
    };
} // namespace akari::render::mlt

namespace akari::render {
    void accept_markov_chain_and_splat(mlt::MLTStats &stats, Rng &rng, const mlt::RadianceRecord &proposal,
                                       mlt::MarkovChain &chain, Film &film);
    std::pair<std::vector<mlt::MarkovChain>, double>
    init_markov_chains(MLTConfig config, const Scene<CPU>  &scene,
                       const std::function<Spectrum(ivec2, Allocator<>, const Scene<CPU>  &, Sampler<CPU> &)> &estimator);
} // namespace akari::render