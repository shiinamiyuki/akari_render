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

#include <random>
#include <spdlog/spdlog.h>
#include <akari/render.h>
#include <akari/render_mlt.h>
#include <numeric>
namespace akari::render {
    Image render_mlt(MLTConfig config, const Scene &scene) {
        using namespace mlt;
        PTConfig pt_config;
        pt_config.max_depth = config.max_depth;
        pt_config.min_depth = config.min_depth;
        auto T = [](const Spectrum &s) { return hmax(s); };
        std::vector<MarkovChain> chains;
        Float b = 0.0;
        {
            std::random_device rd;

            std::vector<uint64_t> seeds;
            seeds.reserve(config.num_bootstrap);
            {
                std::uniform_int_distribution<uint64_t> dist;
                for (int i = 0; i < config.num_bootstrap; i++) {
                    seeds.push_back(dist(rd));
                }
            }

            std::vector<Float> Ts;

            {

                astd::pmr::monotonic_buffer_resource resource;
                for (auto seed : seeds) {
                    Sampler sampler = MLTSampler(seed);
                    sampler.start_next_sample();
                    ivec2 p_film = glm::min(scene.camera->resolution() - 1,
                                            ivec2(sampler.next2d() * vec2(scene.camera->resolution())));
                    auto L = render_pt_pixel_wo_emitter_direct(pt_config, Allocator<>(&resource), scene, sampler,
                                                                     p_film);
                    // spdlog::info("{} {}", p_film[0], p_film[1]);
                    // spdlog::info("{}", T(L));
                    Ts.push_back(T(L));
                }
            }
            Distribution1D distribution(Ts.data(), Ts.size(), Allocator<>());
            std::uniform_real_distribution<> dist;
            astd::pmr::monotonic_buffer_resource resource;
            for (int i = 0; i < config.num_chains; i++) {
                auto [idx, _] = distribution.sample_discrete(dist(rd));
                auto chain = MarkovChain(MLTSampler(seeds[idx]));
                chain.sampler.start_next_sample();
                ivec2 p_film = glm::min(scene.camera->resolution() - 1,
                                        ivec2(chain.sampler.next2d() * vec2(scene.camera->resolution())));
                auto L = render_pt_pixel_wo_emitter_direct(pt_config, Allocator<>(&resource), scene,
                                                                 chain.sampler, p_film);
                AKR_ASSERT(T(L) > 0.0);
                chain.current = RadianceRecord{p_film, L};
                chains.emplace_back(chain);
            }
            b = distribution.integral();
        }
        size_t mutations_per_chain =
            (size_t(config.spp) * size_t(hprod(scene.camera->resolution())) + config.num_chains - 1) /
            size_t(config.num_chains);
        Film film(scene.camera->resolution());
        spdlog::info("{} {}", b, mutations_per_chain);
        std::atomic_uint64_t accepts(0), rejects(0);
        AtomicDouble acc_b(0.0f);
        std::atomic_uint64_t n_large(0);
        thread::parallel_for(config.num_chains, [&](uint32_t id, uint32_t tid) {
            astd::pmr::monotonic_buffer_resource resource;
            auto &chain = chains[id];
            auto &mlt_sampler = *chain.sampler.get<MLTSampler>();
            Rng rng(id);
            mlt_sampler.rng = Rng(rng.uniform_u32());
            for (size_t m = 0; m < mutations_per_chain; m++) {
                chain.sampler.start_next_sample();
                const ivec2 p_film = glm::min(scene.camera->resolution() - 1,
                                              ivec2(chain.sampler.next2d() * vec2(scene.camera->resolution())));
                const auto L = render_pt_pixel_wo_emitter_direct(pt_config, Allocator<>(&resource), scene,
                                                                       chain.sampler, p_film);

                const RadianceRecord proposal{p_film, L};
                const Float accept =
                    T(chain.current.radiance) == 0.0
                        ? 1.0f
                        : std::max<Float>(0.0, std::min<Float>(1.0, T(proposal.radiance) / T(chain.current.radiance)));
                if (mlt_sampler.large_step) {
                    acc_b.add(T(L));
                    n_large++;
                }
                // Float weight1 = (accept + (mlt_sampler.large_step ? 1.0 : 0.0)) /
                // (T(proposal.radiance) / b + mlt_sampler.large_step_prob);
                // Float weight2 = (1 - accept) / (T(chain.current.radiance) / b + mlt_sampler.large_step_prob);
                const Float weight1 = accept / T(proposal.radiance);
                const Float weight2 = (1 - accept) / T(chain.current.radiance);

                // spdlog::info("{} {}", T(L), weight1);
                if (weight1 > 0 && std::isfinite(weight1))
                    film.splat(proposal.p_film, proposal.radiance * weight1 / config.spp);
                if (weight2 > 0 && std::isfinite(weight2))
                    film.splat(chain.current.p_film, chain.current.radiance * weight2 / config.spp);

                if (accept == 1.0 || rng.uniform_float() < accept) {
                    mlt_sampler.accept();
                    accepts++;
                    chain.current = proposal;

                } else {
                    mlt_sampler.reject();
                    rejects++;
                }
                resource.release();
            }
        });
        b = (b * config.num_bootstrap + acc_b.value()) / (config.num_bootstrap + n_large.load());
        spdlog::info("acceptance rate:{}%", accepts * 100 / (accepts + rejects));
        auto array = film.to_array2d();
        array *= Spectrum(b);
        return array2d_to_rgb(array);
    }
} // namespace akari::render