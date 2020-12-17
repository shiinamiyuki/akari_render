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

#include <akari/util.h>
#include <akari/render.h>
#include <spdlog/spdlog.h>

namespace akari::render {
    namespace sms {
        std::pair<Frame, Frame> compute_frame_derivatives(const Vec3 &n, const Vec3 &dpdu, const Vec3 &dndu,
                                                          const Vec3 &dndv) {
            Vec3 s = dpdu - n * dot(n, dpdu);
            Float inv_len_s = 1.0 / length(s);
            s *= inv_len_s;

            Frame dframe_du, dframe_dv;
            dframe_du.s = inv_len_s * (-dndu * dot(n, dpdu) - n * dot(dndu, dpdu));
            dframe_dv.s = inv_len_s * (-dndv * dot(n, dpdu) - n * dot(dndv, dpdu));
            dframe_du.s -= s * dot(dframe_du.s, s);
            dframe_dv.s -= s * dot(dframe_dv.s, s);

            dframe_du.t = cross(dndu, s) + cross(n, dframe_du.s);
            dframe_dv.t = cross(dndv, s) + cross(n, dframe_dv.s);
            dframe_du.n = dndu;
            dframe_dv.n = dndv;
            return std::make_pair(dframe_du, dframe_dv);
        }
        struct ManifoldVertex {
            Vec3 p;
            Vec3 dpdu, dpdv;
            Vec3 n;
            Vec3 dndu, dndv;
            Vec3 s, t;
            Vec3 dsdu, dsdv;
            Vec3 dtdu, dtdv;
            Float eta;
            Vec3 ng;
            Matrix2f A, B, C;
            const MeshInstance *shape = nullptr;
            ManifoldVertex(const SurfaceInteraction &si) {
                p = si.p;
                dpdu = si.dpdu;
                dpdv = si.dpdv;
                n = si.ns;
                dndu = si.dndu;
                dndv = si.dndv;
                ng = si.ng;
                auto frame = Frame(n, dpdu);
                s = frame.s;
                t = frame.t;
                auto [du, dv] = compute_frame_derivatives(n, dpdu, dndu, dndv);
                dsdu = du.s;
                dsdv = dv.s;
                dtdu = du.t;
                dtdv = dv.t;
            }
        };

        struct DirectLighting {
            Ray shadow_ray;
            Spectrum color;
            Float pdf;
        };

        struct SurfaceVertex {
            SurfaceInteraction si;
            Vec3 wo;
            Ray ray;
            Spectrum beta;
            std::optional<BSDF> bsdf;
            Float pdf = 0.0;
            BSDFType sampled_lobe = BSDFType::Unset;
            // SurfaceVertex() = default;
            SurfaceVertex(const Vec3 &wo, const SurfaceInteraction si) : wo(wo), si(si) {}
            Vec3 p() const { return si.p; }
            Vec3 ng() const { return si.ng; }
        };
        struct PathVertex : Variant<SurfaceVertex> {
            using Variant::Variant;
            Vec3 p() const {
                return dispatch([](auto &&arg) { return arg.p(); });
            }
            Vec3 ng() const {
                return dispatch([](auto &&arg) { return arg.ng(); });
            }
            Float pdf() const {
                return dispatch([](auto &&arg) { return arg.pdf; });
            }
            BSDFType sampled_lobe() const {
                return dispatch([=](auto &&arg) -> BSDFType {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, SurfaceVertex>) {
                        return arg.sampled_lobe;
                    }
                    return BSDFType::Unset;
                });
            }
            const Light *light(const Scene *scene) const {
                return dispatch([=](auto &&arg) -> const Light * {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, SurfaceVertex>) {
                        return arg.si.light();
                    }
                    return nullptr;
                });
            }
        };
        struct ManifoldPathSampler {
            const Scene *scene = nullptr;
            Sampler *sampler = nullptr;

            static std::optional<std::pair<Vec2, Vec2>> compute_step(const Vec3 &v0, const ManifoldVertex &v1,
                                                                     const Vec3 &v2, const Vec3 &n_offset) {
                Vec3 wo = v2 - v1.p;
                Float ilo = length(wo);
                if (ilo < 1e-3) {
                    return std::nullopt;
                }
                ilo = 1.0 / ilo;
                wo /= ilo;
                Vec3 wi = v0 - v1.p;
                Float ili = length(wi);
                if (ili < 1e-3) {
                    return std::nullopt;
                }
                wi /= ili;
                Float eta = v1.eta;
                if (dot(wi, v1.ng) < 0.0) {
                    eta = 1.0 / eta;
                }
                Vec3 h = wi + eta * wo;
                if (eta != 1.0) {
                    eta *= 1.0;
                }
                Float ilh = 1.0 / length(h);
                h *= ilh;

                ilo *= eta * ilh;
                ili *= ilh;
                Vec3 dh_du, dh_dv;
                dh_du = -v1.dpdu * (ili + ilo) + wi * (dot(wi, v1.dpdu) * ili) + wo * (dot(wo, v1.dpdu) * ilo);
                dh_dv = -v1.dpdv * (ili + ilo) + wi * (dot(wi, v1.dpdv) * ili) + wo * (dot(wo, v1.dpdv) * ilo);

                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                if (eta != 1.0f) {
                    dh_du *= -1.0f;
                    dh_dv *= -1.0f;
                }
                Matrix2f dH_dX;
                dH_dX(0, 0) = dot(v1.dsdu, h) + dot(v1.s, dh_du);
                dH_dX(1, 0) = dot(v1.dtdu, h) + dot(v1.t, dh_du);
                dH_dX(0, 1) = dot(v1.dsdv, h) + dot(v1.s, dh_dv);
                dH_dX(1, 1) = dot(v1.dtdv, h) + dot(v1.t, dh_dv);
                Float det = dH_dX.determinant();
                if (abs(det) < 1e-6) {
                    return std::nullopt;
                }
                Matrix2f dX_dH = dH_dX.inverse();
                Vec2 H(dot(v1.s, h), dot(v1.t, h));
                Vec2 N(n_offset[0], n_offset[1]);
                Vec2 dH = H - N;
                Vec2 dX = dX_dH * dH;
                return std::make_pair(dH, dX);
            }
            static Float geometric_term(const ManifoldVertex &v0, const ManifoldVertex &v1, const ManifoldVertex &v2) {
                Vec3 wi = v0.p - v1.p;
                Float ili = length(wi);
                if (ili < 1e-3f) {
                    return 0.f;
                }
                ili = 1.0 / (ili);
                wi *= ili;

                Vec3 wo = v2.p - v1.p;
                Float ilo = length(wo);
                if (ilo < 1e-3f) {
                    return 0.f;
                }
                ilo = 1.0 / (ilo);
                wo *= ilo;
                Mat2 dc1_dx0, dc2_dx1, dc2_dx2;
                Float eta = v1.eta;
                if (dot(wi, v1.ng) < 0.0) {
                    eta = 1.0 / eta;
                }
                Vec3 h = wi + eta * wo;
                if (eta != 1.0) {
                    eta *= 1.0;
                }
                Float ilh = 1.0 / length(h);
                h *= ilh;

                ilo *= eta * ilh;
                ili *= ilh;
                Float dot_dpdu_n = dot(v1.dpdu, v1.n), dot_dpdv_n = dot(v1.dpdv, v1.n);
                Vec3 s = v1.dpdu - dot_dpdu_n * v1.n, t = v1.dpdv - dot_dpdv_n * v1.n;

                Vec3 dh_du, dh_dv;
                dh_du = -v1.dpdu * (ili + ilo) + wi * (dot(wi, v1.dpdu) * ili) + wo * (dot(wo, v1.dpdu) * ilo);
                dh_dv = -v1.dpdv * (ili + ilo) + wi * (dot(wi, v1.dpdv) * ili) + wo * (dot(wo, v1.dpdv) * ilo);

                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                if (eta != 1.0f) {
                    dh_du *= -1.0f;
                    dh_dv *= -1.0f;
                }
                Float dot_h_n = dot(h, v1.n), dot_h_dndu = dot(h, v1.dndu), dot_h_dndv = dot(h, v1.dndv);
                Matrix2f dc1_dx1 = Matrix2f(dot(dh_du, s) - dot(v1.dpdu, v1.dndu) * dot_h_n - dot_dpdu_n * dot_h_dndu,
                                            dot(dh_dv, s) - dot(v1.dpdu, v1.dndv) * dot_h_n - dot_dpdu_n * dot_h_dndv,
                                            dot(dh_du, t) - dot(v1.dpdv, v1.dndu) * dot_h_n - dot_dpdv_n * dot_h_dndu,
                                            dot(dh_dv, t) - dot(v1.dpdv, v1.dndv) * dot_h_n - dot_dpdv_n * dot_h_dndv);
                dh_du = ilo * (v2.dpdu - wo * dot(wo, v2.dpdu));
                dh_dv = ilo * (v2.dpdv - wo * dot(wo, v2.dpdv));
                dh_du -= h * dot(dh_du, h);
                dh_dv -= h * dot(dh_dv, h);
                if (eta != 1.f) {
                    dh_du *= -1.f;
                    dh_dv *= -1.f;
                }
                Mat2 dc1_dx2(dot(dh_du, s), dot(dh_dv, s), dot(dh_du, t), dot(dh_dv, t));
                Float determinant = dc1_dx1.determinant();
                if (abs(determinant) < 1e-6f) {
                    return 0.0f;
                }
                Matrix2f inv_dc1_dx1 = dc1_dx1.inverse();
                Float dx1_dx2 = abs((inv_dc1_dx1 * dc1_dx2).determinant());
                // see https://github.com/tizian/specular-manifold-sampling/blob/master/src/librender/manifold_ss.cpp
                /* Unfortunately, these geometric terms are very unstable, so to avoid
                   severe variance we need to clamp here. */
                dx1_dx2 = min(dx1_dx2, Float(1.f));
                Vec3 d = v0.p - v1.p;
                Float inv_r2 = 1.0 / dot(d, d);
                d *= sqrt(inv_r2);
                Float dw0_dx1 = std::abs(dot(d, v1.ng)) * inv_r2;
                Float G = dw0_dx1 * dx1_dx2;
                return G;
            }
            std::optional<SurfaceInteraction> newton_solver(const SurfaceInteraction &si,
                                                            const ManifoldVertex &vtx_init, const Vec3 &light_p) {
                ManifoldVertex vtx = vtx_init;
                std::optional<SurfaceInteraction> solution;
                size_t iter = 0;
                const size_t max_iter = 20;
                const Float threshold = 1e-5;
                const Float step_scale = 1.0;
                Float beta = 1.0f;
                bool success = false;
                while (iter < max_iter) {
                    auto step = compute_step(si.p, vtx, light_p, Vec3(0));
                    if (!step) {
                        break;
                    }
                    auto [C, dX] = *step;
                    if (length(C) < threshold) {
                        success = true;
                        break;
                    }
                    // update v
                    Vec3 p = vtx.p - step_scale * (vtx.dpdu * dX[0] + vtx.dpdv * dX[1]);
                    Vec3 d = normalize(p - si.p);
                    // project to surface

                    Ray ray(si.p, d);
                    solution = scene->intersect(ray);
                    if (!solution || solution->shape != vtx.shape) {
                        beta *= 0.5;
                        iter++;
                        continue;
                    }
                    beta = std::min<Float>(1.0, 2.0 * beta);
                    vtx = ManifoldVertex(*solution);
                    iter++;
                }
                if (!success) {
                    return std::nullopt;
                }
                Vec3 wx = normalize(si.p - vtx.p);
                Vec3 wy = normalize(light_p - vtx.p);
                bool refraction = dot(vtx.ng, wx) * dot(vtx.ng, wy) < 0;
                bool reflection = !refraction;
                if ((vtx.eta == 1.0 && !reflection) || (vtx.eta != 1.0 && !refraction)) {
                    return std::nullopt;
                }
                return solution;
            }
        };
        // Basic Path Tracing
        class SMSPathTracer {
          public:
            const Scene *scene = nullptr;
            Sampler *sampler = nullptr;
            Spectrum L;
            Spectrum beta = Spectrum(1.0f);
            Allocator<> allocator;
            int depth = 0;
            int min_depth = 5;
            int max_depth = 5;

            static Float mis_weight(Float pdf_A, Float pdf_B) {
                pdf_A *= pdf_A;
                pdf_B *= pdf_B;
                return pdf_A / (pdf_A + pdf_B);
            }
            CameraSample camera_ray(const Camera *camera, const ivec2 &p) noexcept {
                CameraSample sample = camera->generate_ray(sampler->next2d(), sampler->next2d(), p);
                return sample;
            }
            std::pair<const Light *, Float> select_light() noexcept {
                return scene->light_sampler->sample(sampler->next2d());
            }

            std::optional<DirectLighting>
            compute_direct_lighting(SurfaceVertex &vertex, const std::pair<const Light *, Float> &selected) noexcept {
                auto [light, light_pdf] = selected;
                if (light) {
                    auto &si = vertex.si;
                    DirectLighting lighting;
                    LightSampleContext light_ctx;
                    light_ctx.u = sampler->next2d();
                    light_ctx.p = si.p;
                    LightSample light_sample = light->sample_incidence(light_ctx);
                    if (light_sample.pdf <= 0.0)
                        return std::nullopt;
                    light_pdf *= light_sample.pdf;
                    auto f = light_sample.I * vertex.bsdf->evaluate(vertex.wo, light_sample.wi) *
                             std::abs(dot(si.ns, light_sample.wi));
                    Float bsdf_pdf = vertex.bsdf->evaluate_pdf(vertex.wo, light_sample.wi);
                    lighting.color = f / light_pdf * mis_weight(light_pdf, bsdf_pdf);
                    lighting.shadow_ray = light_sample.shadow_ray;
                    lighting.pdf = light_pdf;
                    return lighting;
                } else {
                    return std::nullopt;
                }
            }

            void on_miss(const Ray &ray, const std::optional<PathVertex> &prev_vertex) noexcept {
                // if (scene->envmap) {
                //     on_hit_light(scene->envmap.get(), -ray.d, ShadingPoint(), prev_vertex);
                // }
            }

            void accumulate_radiance(const Spectrum &r) { L += r; }

            void on_hit_light(const Light *light, const Vec3 &wo, const ShadingPoint &sp,
                              const std::optional<PathVertex> &prev_vertex) {
                Spectrum I = beta * light->Le(wo, sp);
                if (depth == 0 || BSDFType::Unset != (prev_vertex->sampled_lobe() & BSDFType::Specular)) {
                    accumulate_radiance(I);
                } else {
                    PointGeometry ref;
                    ref.n = prev_vertex->ng();
                    ref.p = prev_vertex->p();
                    auto light_pdf = light->pdf_incidence(ref, -wo) * scene->light_sampler->pdf(light);
                    if ((prev_vertex->sampled_lobe() & BSDFType::Specular) != BSDFType::Unset) {
                        accumulate_radiance(I);
                    } else {
                        Float weight_bsdf = mis_weight(prev_vertex->pdf(), light_pdf);
                        accumulate_radiance(weight_bsdf * I);
                    }
                }
            }
            void accumulate_beta(const Spectrum &k) { beta *= k; }
            // @param mat_pdf: supplied if material is already chosen
            std::optional<SurfaceVertex> on_surface_scatter(const Vec3 &wo, SurfaceInteraction &si,
                                                            const std::optional<PathVertex> &prev_vertex) noexcept {
                auto *material = si.material();
                if (si.triangle.light) {
                    on_hit_light(si.triangle.light, wo, si.sp(), prev_vertex);
                    return std::nullopt;
                } else if (depth < max_depth) {
                    SurfaceVertex vertex(wo, si);
                    auto bsdf = material->evaluate(*sampler, allocator, si);
                    BSDFSampleContext sample_ctx{sampler->next1d(), sampler->next2d(), wo};
                    auto sample = bsdf.sample(sample_ctx);
                    if (!sample) {
                        return std::nullopt;
                    }
                    AKR_ASSERT(sample->pdf >= 0.0f);
                    if (sample->pdf == 0.0f) {
                        return std::nullopt;
                    }
                    vertex.bsdf = bsdf;
                    vertex.ray = Ray(si.p, sample->wi, Eps / std::abs(glm::dot(si.ng, sample->wi)));
                    vertex.beta = sample->f * std::abs(glm::dot(si.ns, sample->wi)) / sample->pdf;
                    vertex.pdf = sample->pdf;
                    return vertex;
                }
                return std::nullopt;
            }
            void run_megakernel(const Camera *camera, const ivec2 &p) noexcept {
                auto camera_sample = camera_ray(camera, p);
                Ray ray = camera_sample.ray;
                std::optional<PathVertex> prev_vertex;
                while (true) {
                    auto si = scene->intersect(ray);
                    if (!si) {
                        on_miss(ray, prev_vertex);
                        break;
                    }
                    auto wo = -ray.d;
                    auto vertex = on_surface_scatter(wo, *si, prev_vertex);
                    if (!vertex) {
                        break;
                    }
                    if ((vertex->sampled_lobe & BSDFType::Specular) == BSDFType::Unset) {
                        std::optional<DirectLighting> has_direct = compute_direct_lighting(*vertex, select_light());
                        if (has_direct) {
                            auto &direct = *has_direct;
                            if (!is_black(direct.color) && !scene->occlude(direct.shadow_ray)) {
                                accumulate_radiance(beta * direct.color);
                            }
                        }
                    }
                    accumulate_beta(vertex->beta);
                    depth++;
                    if (depth > min_depth) {
                        Float continue_prob = std::min<Float>(1.0, hmax(beta)) * 0.95;
                        if (continue_prob < sampler->next1d()) {
                            accumulate_beta(Spectrum(1.0 / continue_prob));
                        } else {
                            break;
                        }
                    }
                    ray = vertex->ray;
                    prev_vertex = PathVertex(*vertex);
                }
            }
        };
    } // namespace sms
    Film render_sms(SMSConfig config, const Scene &scene) {
        Film film(scene.camera->resolution());
        std::vector<astd::pmr::monotonic_buffer_resource*> buffers;
        for (size_t i = 0; i < thread::num_work_threads(); i++) {
            buffers.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::new_delete_resource()));
        }
        thread::parallel_for(thread::blocked_range<2>(film.resolution(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            auto Li = [&](const ivec2 p, Sampler &sampler) -> Spectrum {
                sms::SMSPathTracer pt;
                pt.min_depth = config.min_depth;
                pt.max_depth = config.max_depth;
                pt.L = Spectrum(0.0);
                pt.beta = Spectrum(1.0);
                pt.sampler = &sampler;
                pt.scene = &scene;
                pt.allocator = Allocator<>(buffers[tid]);
                pt.run_megakernel(&scene.camera.value(), p);
                buffers[tid]->release();
                return pt.L;
            };
            Sampler sampler = config.sampler;
            sampler.set_sample_index(id.y * film.resolution().x + id.x);
            for (int s = 0; s < config.spp; s++) {
                auto L = Li(id, sampler);
                film.add_sample(id, L, 1.0);
            }
        });
        for(auto buf: buffers){
            delete buf;
        }
        spdlog::info("render sms pt done");
        return film;
    }
} // namespace akari::render