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

#include <unordered_map>
#include <random>
#include <spdlog/spdlog.h>
#include <akari/render.h>
#include <akari/render_mlt.h>
#include <numeric>
namespace akari::render {
    struct IVec2Hash {
        size_t operator()(const ivec2 &p) const {
            auto h = std::hash<size_t>();
            return h(astd::bit_cast<size_t>(p));
        }
    };
    struct IVec2Equal {
        bool operator()(const ivec2 &p, const ivec2 &q) const {
            return astd::bit_cast<size_t>(p) == astd::bit_cast<size_t>(q);
        }
    };
    namespace smcmc {
        using namespace mlt;
        struct TileEstimator : std::array<RadianceRecord, 5> {};
        struct CoherentSamples : std::array<RadianceRecord, 5> {};
        struct Tile {
            ivec2 p_center;
            std::optional<Sampler> sampler;
            TileEstimator mcmc_estimate, mc_estimate;
            CoherentSamples current;
            uint32_t n_mc_estimates = 0, n_mcmc_estimates = 0;
            double b = 0.0;
        };
    } // namespace smcmc
    Image render_smcmc(MLTConfig config, const Scene &scene) {
        using namespace mlt;
        using namespace smcmc;
        PTConfig pt_config;
        pt_config.max_depth = config.max_depth;
        pt_config.min_depth = config.min_depth;
        auto T = [](const Spectrum &s) -> Float {
            AKR_ASSERT(hmax(s) >= 0.0);
            return hmax(s);
        };
        const std::array<ivec2, 5> offsets = {
            ivec2(0, 0), ivec2(0, 1), ivec2(0, -1), ivec2(1, 0), ivec2(-1, 0),
        };
        std::unordered_map<ivec2, int, IVec2Hash, IVec2Equal> offset2index;
        for (int i = 0; i < 5; i++) {
            offset2index[offsets[i]] = i;
        }
        astd::pmr::monotonic_buffer_resource _buffer(astd::pmr::new_delete_resource());
        astd::pmr::vector<ivec2> overlapped_tile_offsets((Allocator<>(&_buffer)));
        std::unordered_map<ivec2, astd::pmr::vector<int>, IVec2Hash, IVec2Equal,
                           Allocator<std::pair<const ivec2, astd::pmr::vector<int>>>>
            overlapped_pixel_indices((Allocator<std::pair<const ivec2, astd::pmr::vector<int>>>(&_buffer)));

        {
            for (int x = -2; x <= 2; x++) {
                for (int y = -2; y <= 2; y++) {
                    if (x != 0 || y != 0) {
                        astd::pmr::vector<int> indices;
                        ivec2 p(x, y);
                        for (auto &off : offsets) {
                            auto q = p + off;
                            if (offset2index.find(q) != offset2index.end()) {
                                indices.push_back(offset2index.at(q));
                            }
                        }
                        if (!indices.empty()) {
                            overlapped_pixel_indices[p] = std::move(indices);
                            overlapped_tile_offsets.push_back(p);
                        }
                    }
                }
            }
        }
        auto run_uniform_global_mcmc = [&](PTConfig config, Allocator<> allocator, Sampler &sampler) {
            sampler.start_next_sample();
            ivec2 p_center =
                glm::min(scene.camera->resolution() - 1, ivec2(sampler.next2d() * vec2(scene.camera->resolution())));
            auto idx = std::min<int>(sampler.next1d() * 5, 4);
            auto p_film = p_center + offsets[idx];
            auto L = render_pt_pixel_wo_emitter_direct(pt_config, allocator, scene, sampler, p_film);
            return std::make_pair(p_center, L);
        };
        auto run_mcmc = [&](PTConfig config, Allocator<> allocator, const ivec2 &p_center, Sampler &sampler) {
            sampler.start_next_sample();
            (void)sampler.next2d();
            auto idx = std::min<int>(sampler.next1d() * 5, 4);
            auto p_film = p_center + offsets[idx];
            auto L = render_pt_pixel_wo_emitter_direct(pt_config, allocator, scene, sampler, p_film);
            return std::make_pair(idx, L);
        };
        auto run_mcmc_coherent = [&](PTConfig config, Allocator<> allocator, const ivec2 &p_center,
                                     const MLTSampler &base, int idx) {
            astd::pmr::vector<Float> Xs(allocator);
            for (auto &X : base.X) {
                Xs.push_back(X.value);
            }
            Sampler sampler = ReplaySampler(std::move(Xs), base.rng);
            sampler.start_next_sample();
            (void)sampler.next2d();
            (void)sampler.next1d();
            auto p_film = p_center + offsets[idx];
            auto L = render_pt_pixel_wo_emitter_direct(pt_config, allocator, scene, sampler, p_film);
            return L;
        };
        std::optional<MarkovChain> global_chain;
        std::random_device rd;
        {
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
                    auto [p_film, L] = run_uniform_global_mcmc(pt_config, Allocator<>(&resource), sampler);
                    Ts.push_back(T(L));
                }
            }
            Distribution1D distribution(Ts.data(), Ts.size(), Allocator<>());
            std::uniform_real_distribution<> dist;
            global_chain = MarkovChain(MLTSampler(seeds[distribution.sample_discrete(dist(rd)).first]));
        }

        Array2D<Tile> tiles(scene.camera->resolution());
        thread::parallel_for(thread::blocked_range<2>(tiles.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            auto &s = tiles(id);
            for (int i = 0; i < 5; i++) {
                s.mcmc_estimate[i].p_film = id + offsets[i];
                s.mc_estimate[i].p_film = id + offsets[i];
                s.p_center = id;
            }
        });
        {
            std::uniform_int_distribution<uint32_t> dist;
            astd::pmr::monotonic_buffer_resource resource;
            for (int i = 0; i < hprod(scene.camera->resolution()); i++) {
                auto [p_film, L] = run_uniform_global_mcmc(pt_config, Allocator<>(&resource), global_chain->sampler);
                if (!tiles(p_film).sampler.has_value()) {
                    tiles(p_film).sampler = global_chain->sampler;
                    tiles(p_film).sampler->get<MLTSampler>()->rng = Rng(dist(rd));
                    tiles(p_film).p_center = p_film;
                }
            }
        }
        auto unscaled_mcmc_estimator = [&](const Tile &state, const ivec2 &p) -> Spectrum {
            // for (int i = 0; i < 5; i++) {
            //     if (glm::all(glm::equal(state.p_center + offsets[i], p))) {
            //         return sample[i].radiance;
            //     }
            // }
            // AKR_ASSERT(false);
            auto off = p - state.p_center;
            int i = offset2index.at(off);
            return state.mcmc_estimate[i].radiance / state.n_mcmc_estimates;
        };
        auto mc_estimator = [&](const Tile &state, const ivec2 &p) -> Spectrum {
            // for (int i = 0; i < 5; i++) {
            //     if (glm::all(glm::equal(state.p_center + offsets[i], p))) {
            //         return sample[i].radiance;
            //     }
            // }
            // AKR_ASSERT(false);
            auto off = p - state.p_center;
            int i = offset2index.at(off);
            if (state.n_mc_estimates == 0) {
                return Spectrum(0.0);
            }
            return state.mc_estimate[i].radiance / state.n_mc_estimates;
        };
        auto estimator = [&](const Tile &state, const CoherentSamples &sample) -> TileEstimator {
            TileEstimator G;
            for (auto &X : sample) {
                for (int i = 0; i < 5; i++) {
                    if (glm::all(glm::equal(X.p_film - state.p_center, offsets[i]))) {
                        G[i] = X;
                    }
                }
            }
            return G;
        };
        auto run_mcmc2 = [&](Allocator<> alloc, Tile &state) -> CoherentSamples {
            CoherentSamples Xs;
            auto [X_idx, X] = run_mcmc(pt_config, alloc, state.p_center, *state.sampler);
            Xs[0] = RadianceRecord{state.p_center + offsets[X_idx], X};
            int cnt = 1;
            for (int i = 0; i < 5; i++) {
                if (i != X_idx) {
                    Xs[cnt] = RadianceRecord{
                        state.p_center + offsets[i],
                        run_mcmc_coherent(pt_config, alloc, state.p_center, *state.sampler->get<MLTSampler>(), i)};
                    cnt++;
                }
            }
            return Xs;
        };
        auto Ts = [&](const CoherentSamples &Xs) {
            auto value = T(Xs[0].radiance);
            for (int i = 1; i < 5; i++) {
                value = std::max(value, T(Xs[i].radiance));
            }
            return value;
        };
        // ensures every tile is initialized
        thread::parallel_for(thread::blocked_range<2>(tiles.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            std::uniform_int_distribution<uint32_t> dist;
            astd::pmr::monotonic_buffer_resource resource;
            if (!tiles(id).sampler.has_value()) {
                const int num_tries = 16;
                for (int i = 0; i < num_tries; i++) {
                    Sampler sampler = MLTSampler(dist(rd));
                    auto [idx, L] = run_mcmc(pt_config, Allocator<>(&resource), id, sampler);
                    if (T(L) > 0.0 || i == num_tries - 1) {
                        tiles(id).sampler = sampler;
                        tiles(id).sampler->get<MLTSampler>()->rng = Rng(dist(rd));
                        tiles(id).p_center = id;
                        break;
                    }
                }
            }
            AKR_ASSERT(tiles(id).sampler.has_value());
            auto L = run_mcmc2(Allocator<>(&resource), tiles(id));
            tiles(id).current = L;
        });
        AtomicDouble acc_b(0.0);
        std::atomic_uint64_t n_large(0);
        auto splat = [&](Tile &a, const CoherentSamples &Xs, Float weight) {
            auto G = estimator(a, Xs);
            for (int i = 0; i < 5; i++) {
                AKR_ASSERT(glm::all(glm::equal(G[i].p_film, a.mcmc_estimate[i].p_film)));
                a.mcmc_estimate[i].radiance += G[i].radiance * weight;
            }
        };
        auto splat_mc = [&](Tile &a, const CoherentSamples &Xs, Float weight) {
            auto G = estimator(a, Xs);
            for (int i = 0; i < 5; i++) {
                AKR_ASSERT(glm::all(glm::equal(G[i].p_film, a.mc_estimate[i].p_film)));
                a.mc_estimate[i].radiance += G[i].radiance * weight;
            }
        };
        auto replica_exchange = [&](Allocator<> alloc, Tile &a, Tile &b) {
            std::uniform_real_distribution<Float> dist;
            Tile ab;
            ab.p_center = a.p_center;
            ab.sampler = b.sampler;
            Tile ba;
            ba.p_center = b.p_center;
            ba.sampler = a.sampler;
            const auto Lab = run_mcmc2(alloc, ab);
            const auto Lba = run_mcmc2(alloc, ba);
            const auto Tab = Ts(Lab);
            const auto Tba = Ts(Lba);
            const auto Ta = Ts(a.current);
            const auto Tb = Ts(b.current);
            const auto accept = std::clamp<Float>(Tab * Tba / (Ta * Tb), 0.0, 1.0);

            if (accept > 0) {
                splat(ab, Lab, accept / Tab);
                splat(ba, Lba, accept / Tba);
            }
            if (1.0 - accept > 0) {
                splat(a, a.current, (1.0 - accept) / Ta);
                splat(b, b.current, (1.0 - accept) / Tb);
            }

            if (dist(rd) < accept) {
                a = ab;
                b = ba;

                a.current = Lab;
                b.current = Lba;
            }
            a.n_mcmc_estimates++;
            b.n_mcmc_estimates++;
        };
        auto independent_mcmc = [&](Allocator<> alloc, Tile &s) {
            std::uniform_real_distribution<Float> dist;
            const auto L = run_mcmc2(alloc, s);
            const auto Tnew = Ts(L);
            const auto accept = std::clamp<Float>(Tnew / Ts(s.current), 0.0, 1.0);
            auto mlt_sampler = s.sampler->get<MLTSampler>();
            if (mlt_sampler->large_step) {
                acc_b.add(Ts(L));
                n_large++;
                splat_mc(s, L, 1.0);
                s.n_mc_estimates++;
            }
            if (accept > 0) {
                // film.splat(s.p_center, estimator(s, L) * accept / Tnew);
                splat(s, L, accept / Tnew);
            }
            if (1.0 - accept > 0) {
                // film.splat(s.p_center, s.current.radiance * (1.0 - accept) / s.current.Ts);
                splat(s, s.current, (1.0 - accept) / Ts(s.current));
            }
            if (dist(rd) < accept) {
                mlt_sampler->accept();
                s.current = L;
            } else {
                mlt_sampler->reject();
            }
            s.n_mcmc_estimates++;
        };
        // run mcmc

        std::vector<astd::pmr::monotonic_buffer_resource *> buffers;
        for (size_t i = 0; i < thread::num_work_threads(); i++) {
            buffers.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::new_delete_resource()));
        }

        for (int m = 0; m < config.spp; m++) {
            // if (m % 2 == 0) {
            thread::parallel_for(
                thread::blocked_range<2>(tiles.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
                    independent_mcmc(Allocator<>(buffers[tid]), tiles(id)); //                           .
                    buffers[tid]->release();
                });
            // } else {
            //     int n = m / 2;
            //     switch (n % 4) {
            //     case 0:
            //         break;
            //     case 1:
            //         break;
            //     case 2:
            //         break;
            //     case 3:
            //         break;
            //     }
            // }
        }
        const auto b = acc_b.value() / n_large.load();

        const auto alpha = 0.05;
        const auto beta1 = 0.05, beta2 = 0.5;
        auto in_tile = [&](const Tile &s, ivec2 p) -> bool {
            auto off = p - s.p_center;
            return offset2index.find(off) != offset2index.end();
        };
        auto for_each_overlapped_tile = [&](const Tile &s, auto &&F) {
            for (auto &off : overlapped_tile_offsets) {
                auto id = off + s.p_center;
                if (glm::any(glm::lessThan(id, ivec2(0))) || glm::any(glm::greaterThanEqual(id, tiles.dimension()))) {
                    continue;
                }
                F(tiles(id));
            }
        };

        auto for_each_overlapped_pixel = [&](const Tile &s, const Tile &t, auto &&F) {
            auto t_off = t.p_center - s.p_center;
            if (overlapped_pixel_indices.find(t_off) == overlapped_pixel_indices.end())
                return;
            for (auto &i : overlapped_pixel_indices.at(t_off)) {
                F(s.p_center + offsets[i]);
            }
        };
        auto for_each_overlapped_tile_pixel = [&](const Tile &s, auto &&F) {
            for (auto &off : overlapped_tile_offsets) {
                auto id = off + s.p_center;
                if (glm::any(glm::lessThan(id, ivec2(0))) || glm::any(glm::greaterThanEqual(id, tiles.dimension()))) {
                    continue;
                }
                auto &t = tiles(id);
                for (auto &i : overlapped_pixel_indices.at(off)) {
                    F(t, s.p_center + offsets[i]);
                }
            }
        };
        auto average_tile_unscaled_mcmc = [&](const Tile &s) {
            double avg = 0.0;
            for (auto &r : s.mcmc_estimate) {
                avg += T(r.radiance);
            }
            avg /= s.mcmc_estimate.size();
            if (s.n_mcmc_estimates != 0)
                avg /= s.n_mcmc_estimates;
            return avg;
        };
        auto average_tile_mcmc = [&](const Tile &s) { return average_tile_unscaled_mcmc(s) * s.b; };
        auto average_tile_mc = [&](const Tile &s) {
            double avg = 0.0;
            for (auto &r : s.mc_estimate) {
                avg += T(r.radiance);
            }
            avg /= s.mc_estimate.size();
            if (s.n_mc_estimates != 0)
                avg /= s.n_mc_estimates;
            return avg;
        };
        const auto Fst = [&](const Tile &s, const Tile &t, const ivec2 &k) {
            return 0.5 * (T(unscaled_mcmc_estimator(t, k)) * t.b - T(unscaled_mcmc_estimator(s, k)) * s.b);
        };
        const auto e = [&](const Tile &s) {
            auto val = alpha * std::abs(average_tile_mcmc(s) - average_tile_mc(s));
            // for_each_overlapped_tile(s, [&](const Tile &t) {
            //     for_each_overlapped_pixel(s, t, [&](const ivec2 &k) { val += std::abs(Fst(s, t, k)); });
            // });
            for_each_overlapped_tile_pixel(s, [&](const Tile &t, const ivec2 &k) { val += std::abs(Fst(s, t, k)); });
            return val;
        };

        // reconstruction
        spdlog::info("reconstructing...");
        spdlog::info("b={}", b);
        auto output = rgb_image(tiles.dimension());
        Array2D<double> error(tiles.dimension());
        error.fill(0);
        thread::parallel_for(thread::blocked_range<2>(tiles.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            auto &s = tiles(id);
            s.b = average_tile_mc(s);
        });
        auto bs = rgb_image(tiles.dimension());
        auto mcmc = rgb_image(tiles.dimension());
        auto inc_bs = rgb_image(tiles.dimension());
        Array2D<double> new_b(tiles.dimension());
        new_b.fill(b);
        for (int n = 0; n < 1000; n++) {
            if (n % 50 == 0) {
                spdlog::info("updating error");
                thread::parallel_for( //
                    thread::blocked_range<2>(tiles.dimension(), ivec2(16, 16)),
                    [&](ivec2 id, uint32_t tid) { error(id) = e(tiles(id)); });
            }
            thread::parallel_for( //
                thread::blocked_range<2>(tiles.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
                    const auto w1 = [&](const Tile &s) {
                        return 1.0 / (error(s.p_center) + beta1 * std::pow(beta2, n));
                    };
                    const auto w2 = [&](const Tile &s, const Tile &t) { return std::min(w1(s), w1(t)); };
                    auto &s = tiles(id);
                    double inc_b = 0.0;
                    double inc_b_denom = 0.0;
                    inc_b += alpha * w1(s) * (average_tile_mc(s) - average_tile_mcmc(s));
                    inc_b_denom += alpha * w1(s) * average_tile_unscaled_mcmc(s);

                    for_each_overlapped_tile_pixel(s, [&](const Tile &t, const ivec2 &k) {
                        const auto w_st = w2(s, t);
                        inc_b += w_st * Fst(s, t, k);
                        inc_b_denom += w_st * T(unscaled_mcmc_estimator(t, k));
                    });
                    inc_bs(id, 0) = inc_b;
                    inc_b /= inc_b_denom;
                    if (std::isfinite(inc_b)) {
                        // printf("%f %f\n", inc_b * inc_b_denom, inc_b_denom);
                        new_b(id) = s.b + inc_b;
                    } else {
                        new_b(id) = s.b;
                    }
                    new_b(id) = std::max(new_b(id), 0.0);
                    inc_bs(id, 1) = inc_b_denom;
                });
            if (n % 50 == 0)
                spdlog::info("iter={}", n);
            thread::parallel_for( //
                thread::blocked_range<2>(tiles.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
                    auto &s = tiles(id);
                    s.b = new_b(id);
                    for (int i = 0; i < 3; i++) {
                        bs(id, i) = s.b;
                        mcmc(id, i) = average_tile_mcmc(s);
                    }
                });
            // write_hdr(inc_bs, fmt::format("inc_b{}.exr", n));
            // write_hdr(bs, fmt::format("b{}.exr", n));
            // write_hdr(mcmc, fmt::format("mcmc{}.exr", n));
        }

        auto unscaled = rgb_image(tiles.dimension());
        auto mc = rgb_image(tiles.dimension());
        thread::parallel_for( //
            thread::blocked_range<2>(tiles.dimension(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
                auto &s = tiles(id);
                auto I = unscaled_mcmc_estimator(s, s.p_center) * s.b;
                for (int i = 0; i < 3; i++) {
                    output(id, i) = I[i];
                    unscaled(id, i) = unscaled_mcmc_estimator(s, s.p_center)[i];
                    mc(id, i) = mc_estimator(s, s.p_center)[i];
                };
            });
        write_hdr(unscaled, "unscaled.exr");
        write_hdr(mc, "mc.exr");
        for (auto buf : buffers) {
            delete buf;
        }
        spdlog::info("render smcmc done");
        return output;
    }
} // namespace akari::render