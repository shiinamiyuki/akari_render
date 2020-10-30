// MIT License
//
// Copyright (c) 2020 椎名深雪
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <mutex>
#include <stack>
#include <akari/core/parallel.h>
#include <akari/core/profiler.h>
#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/common.h>
#include <akari/render/pathtracer.h>

namespace akari::render {
    class QTreeNode {
      public:
        std::array<AtomicFloat, 4> _sum;
        std::array<int, 4> _children = {-1, -1, -1, -1};

        QTreeNode() { setSum(0); }

        void copyFrom(const QTreeNode &node) {
            _children = (node._children);
            for (int i = 0; i < 4; i++) {
                _sum[i].set(float(node._sum[i]));
            }
        }

        QTreeNode(const QTreeNode &node) { copyFrom(node); }

        QTreeNode &operator=(const QTreeNode &node) {
            if (&node == this) {
                return *this;
            }
            copyFrom(node);
            return *this;
        }

        void setSum(Float val) {
            for (int i = 0; i < 4; i++) {
                _sum[i].set(val);
            }
        }

        QTreeNode *child(int i, std::vector<QTreeNode> &nodes) const {
            return _children[i] > 0 ? &nodes[_children[i]] : nullptr;
        }

        static int childIndex(const vec2 &p) {
            int x, y;
            if (p.x < 0.5f) {
                x = 0;
            } else {
                x = 1;
            }
            if (p.y < 0.5f) {
                y = 0;
            } else {
                y = 1;
            }
            return x + 2 * y;
        }

        [[nodiscard]] bool isLeaf(int i) const { return _children[i] <= 0; }

        [[nodiscard]] static vec2 offset(size_t i) { return vec2(i & 1u, i >> 1u) * 0.5f; }

        [[nodiscard]] Float sum() const {
            Float v = 0;
            for (auto &i : _sum) {
                v += (float)i;
            }
            return v;
        }

        float eval(const vec2 &p, std::vector<QTreeNode> &nodes) const {
            auto idx = childIndex(p);
            if (child(idx, nodes)) {
                return 4.0f * child(idx, nodes)->eval((p - offset(idx)) * 2.0f, nodes);
            } else {
                AKR_CHECK(_sum[idx].value() >= 0.0f);
                return 4.0f * float(_sum[idx]);
            }
        }

        float pdf(const vec2 &p, std::vector<QTreeNode> &nodes) const {
            auto idx = childIndex(p);
            //            if (!(_sum[idx].value() > 0)) {
            //                return 0.0f;
            //            }
            auto s = sum();
            AKR_CHECK(s >= 0);
            AKR_CHECK(!std::isnan(s));
            auto factor = s <= 0.0f ? 0.25f : _sum[idx].value() / s;
            if (factor < 0) {
                info("{} {} {}\n", factor, _sum[idx].value(), s);
            }
            AKR_CHECK(factor >= 0);
            if (child(idx, nodes)) {
                return 4.0f * factor * child(idx, nodes)->pdf((p - offset(idx)) * 2.0f, nodes);
            } else {
                return 4.0f * factor;
            }
        }

        vec2 sample(vec2 u, vec2 u2, std::vector<QTreeNode> &nodes) const {
            std::array<float, 4> m = {float(_sum[0]), float(_sum[1]), float(_sum[2]), float(_sum[3])};
            auto left = m[0] + m[2];
            auto right = m[1] + m[3];
            auto total = left + right;
            // log::log("total: {}\n", total);
            //            AKR_CHECK(total > 0);
            if (total == 0) {
                total = 1;
                m[0] = m[1] = m[2] = m[3] = 0.25;
                left = m[0] + m[2];
                right = m[1] + m[3];
            }

            int x, y;
            if (u[0] < left / total) {
                x = 0;
                u[0] /= left / total;
            } else {
                x = 1;
                u[0] = (u[0] - left / total) / (right / total);
            }
            auto up = m[x];
            auto down = m[2 + x];
            total = up + down;
            if (u[1] < up / total) {
                y = 0;
                u[1] /= up / total;
            } else {
                y = 1;
                u[1] = (u[1] - up / total) / (down / total);
            }
            int child = x + 2 * y;
            vec2 sampled;
            if (this->child(child, nodes)) {
                sampled = this->child(child, nodes)->sample(u, u2, nodes);
            } else {
                sampled = u2;
            }
            return vec2(x, y) * 0.5f + sampled * 0.5f;
        }

        void deposit(const vec2 &p, Float e, std::vector<QTreeNode> &nodes) {
            int idx = childIndex(p);
            _sum[idx].add(e);
            auto c = child(idx, nodes);
            AKR_CHECK(e >= 0);
            AKR_CHECK(_sum[idx].value() >= 0);
            if (c) {
                c->deposit((p - offset(idx)) * 2.0f, e, nodes);
            }
        }
    };

    static vec3 canonicalToDir(const vec2 &p) {
        const Float cosTheta = 2 * p.y - 1;
        const Float phi = 2 * Pi * p.x;

        const Float sinTheta = sqrt(1 - cosTheta * cosTheta);
        Float sinPhi, cosPhi;
        sinPhi = sin(phi);
        cosPhi = cos(phi);

        return vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
    }

    static vec2 dirToCanonical(const vec3 &d) {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
            return vec2(0, 0);
        }

        const Float cosTheta = std::min(std::max(d.y, -1.0f), 1.0f);
        Float phi = std::atan2(d.z, d.x);
        while (phi < 0)
            phi += 2.0 * Pi;

        return vec2(phi / (2 * Pi), (cosTheta + 1) / 2);
    }

    class DTree {
      public:
        AtomicFloat sum;
        AtomicFloat weight;
        std::vector<QTreeNode> nodes;

        DTree() : sum(0), weight(0) {
            nodes.emplace_back();
            _build();
        }

        DTree(const DTree &other) {
            sum.set(other.sum.value());
            weight.set(other.weight.value());
            nodes = (other.nodes);
        }

        DTree &operator=(const DTree &other) {
            if (&other == this) {
                return *this;
            }
            sum.set(other.sum.value());
            weight.set(other.weight.value());
            nodes = (other.nodes);
            return *this;
        }

        vec2 sample(const vec2 &u, const vec2 &u2) { return nodes[0].sample(u, u2, nodes); }

        Float pdf(const vec2 &p) { return nodes[0].pdf(p, nodes); }

        Float eval(const vec2 &u) {
            return nodes[0].eval(u, nodes); /// weight.value();
        }

        void _build() {
            auto updateSum = [this](QTreeNode &node, auto &&update) -> double {
                for (int i = 0; i < 4; i++) {
                    if (!node.isLeaf(i)) {
                        auto c = node.child(i, nodes);
                        update(*c, update);
                        node._sum[i].set(c->sum());
                    }
                }
                //    log::log("sum: {}\n",node.sum());
                return node.sum();
            };
            sum.set(updateSum(nodes.front(), updateSum));
        }

        void reset() {
            for (auto &i : nodes) {
                i.setSum(0);
            }
            sum.set(0);
        }

        void refine(const DTree &prev, Float threshold) {
            struct StackNode {
                size_t node = (size_t)-1, otherNode = (size_t)-1;
                const DTree *tree = nullptr;
                int depth;
            };
            std::stack<StackNode> stack;
            nodes.clear();
            nodes.emplace_back();
            stack.push({0, 0, &prev, 1});
            sum.set(0.0f);
            auto total = prev.sum.value();
            //            log::log("{} {}\n", total, threshold);
            while (!stack.empty()) {
                auto node = stack.top();
                stack.pop();

                // log::log("other index: {}, sum: {}\n",node.otherNode, otherNode.sum());
                for (int i = 0; i < 4; ++i) {
                    AKR_CHECK(node.otherNode >= 0);
                    auto &otherNode = node.tree->nodes.at(node.otherNode);
                    auto fraction = total == 0.0f ? std::pow(0.25, node.depth) : otherNode._sum[i].value() / total;
                    // log::log("{} {}\n", otherNode._sum[i].value(), fraction);
                    if (fraction >= threshold && node.depth < 10) {
                        if (otherNode.isLeaf(i)) {
                            stack.push({nodes.size(), nodes.size(), this, node.depth + 1});
                        } else {
                            AKR_CHECK(otherNode._children[i] > 0);
                            AKR_CHECK((size_t)otherNode._children[i] != node.otherNode);
                            AKR_CHECK(node.tree == &prev);
                            stack.push({nodes.size(), (size_t)otherNode._children[i], &prev, node.depth + 1});
                        }
                        nodes[node.node]._children[i] = nodes.size();
                        auto val = otherNode._sum[i].value() / 4.0f;
                        nodes.emplace_back();
                        nodes.back().setSum(val);
                    }
                }
            }
            //           log::log("QTreeNodes: {}\n", nodes.size());
            weight.add(1);
            reset();
        }

        void deposit(const vec2 &p, Float e) {
            if (e <= 0)
                return;
            sum.add(e);
            nodes[0].deposit(p, e, nodes);
        }
    };

    class DTreeWrapper {
      public:
        bool valid = true;
        DTree building, sampling;

        //        DTreeWrapper(){
        //            sampling.nodes[0].setSum(0.25);
        //        }

        vec3 sample(const vec2 &u, const vec2 &u2) { return canonicalToDir(sampling.sample(u, u2)); }

        Float pdf(const vec3 &w) { return sampling.pdf(dirToCanonical(w)) * Inv4Pi; }

        Float eval(const vec3 &w) { return sampling.eval(dirToCanonical(w)); }

        void deposit(const vec3 &w, Float e) {
            AKR_CHECK(!building.nodes.empty());
            auto p = dirToCanonical(w);
            building.deposit(p, e);
        }

        void refine() {
            AKR_CHECK(building.sum.value() >= 0.0f);
            //            building._build();
            sampling = building;
            sampling._build();
            AKR_CHECK(sampling.sum.value() >= 0.0f);
            building.refine(sampling, 0.01);
        }
    };

    class STreeNode {
      public:
        DTreeWrapper dTree;
        std::atomic<int> nSample = 0;
        std::array<int, 2> _children = {-1, -1};
        int axis = 0;
        bool _isLeaf = true;

        STreeNode() = default;

        STreeNode(const STreeNode &other)
            : dTree(other.dTree), nSample((int)other.nSample), _children(other._children), axis(other.axis),
              _isLeaf(other._isLeaf) {}

        [[nodiscard]] bool isLeaf() const { return _isLeaf; }

        vec3 sample(vec3 p, const vec2 &u, const vec2 &u2, std::vector<STreeNode> &nodes) {
            if (isLeaf()) {
                return dTree.sample(u, u2);
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    return nodes[_children[0]].sample(p, u, u2, nodes);
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    return nodes[_children[1]].sample(p, u, u2, nodes);
                }
            }
        }

        Float pdf(vec3 p, const vec3 &w, std::vector<STreeNode> &nodes) {
            if (isLeaf()) {
                return dTree.pdf(w);
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    return nodes.at(_children[0]).pdf(p, w, nodes);
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    return nodes.at(_children[1]).pdf(p, w, nodes);
                }
            }
        }

        auto getDTree(vec3 p, std::vector<STreeNode> &nodes) {
            //            AKR_CHECK(0.0f - 1e-6f <= p[axis] && p[axis] <= 1.0f + 1e-6f);
            //            log::log("{} {}\n",axis,p[axis]);
            if (isLeaf()) {
                return &dTree;
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    return nodes.at(_children[0]).getDTree(p, nodes);
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    return nodes.at(_children[1]).getDTree(p, nodes);
                }
            }
        }

        Float eval(vec3 p, const vec3 &w, std::vector<STreeNode> &nodes) {
            if (isLeaf()) {
                return dTree.eval(w);
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    return nodes.at(_children[0]).eval(p, w, nodes);
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    return nodes.at(_children[1]).eval(p, w, nodes);
                }
            }
        }

        void deposit(vec3 p, const vec3 &w, Float irradiance, std::vector<STreeNode> &nodes) {
            if (isLeaf()) {
                nSample++;
                dTree.deposit(w, irradiance);
            } else {
                if (p[axis] < 0.5f) {
                    p[axis] *= 2.0f;
                    nodes.at(_children[0]).deposit(p, w, irradiance, nodes);
                } else {
                    p[axis] = (p[axis] - 0.5f) * 2.0f;
                    nodes.at(_children[1]).deposit(p, w, irradiance, nodes);
                }
            }
        }
    };

    class STree {
      public:
        std::vector<STreeNode> nodes;

        explicit STree(const Bounds3f &box) : nodes(1) {
            auto sz = hmax(box.size()) * 0.5f;
            auto centroid = box.centroid();
            this->box = Bounds3f{centroid - vec3(sz), centroid + vec3(sz)};
        }

        Bounds3f box;

        vec3 sample(const vec3 &p, const vec2 &u, const vec2 &u2) {
            return nodes.at(0).sample(box.offset(p), u, u2, nodes);
        }

        Float pdf(const vec3 &p, const vec3 &w) { return nodes.at(0).pdf(box.offset(p), w, nodes); }

        auto dTree(const vec3 &p) { return nodes[0].getDTree(box.offset(p), nodes); }

        Float eval(const vec3 &p, const vec3 &w) { return nodes.at(0).eval(box.offset(p), w, nodes); }

        void deposit(vec3 p, const vec3 &w, Float irradiance) {
            if (irradiance >= 0 && !std::isnan(irradiance)) {
                nodes.at(0).deposit(box.offset(p), w, irradiance, nodes);
            }
        }

        void refine(int idx, size_t maxSample, int depth) {
            if (nodes[idx].isLeaf() && (size_t)nodes[idx].nSample > maxSample) {
                //                log::log("sum {}\n", nodes[idx].dTree.building.sum.value());
                //                log::log("samples: {} {}\n", (int) nodes[idx].nSample,maxSample);
                for (int i = 0; i < 2; i++) {
                    nodes[idx]._children[i] = nodes.size();
                    nodes.emplace_back();
                    auto &node = nodes.back();
                    node.nSample = nodes[idx].nSample / 2;
                    node.axis = (nodes[idx].axis + 1) % 3;
                    node.dTree = nodes[idx].dTree;
                    AKR_CHECK(node.isLeaf());
                }
                nodes[idx]._isLeaf = false;
                nodes[idx].dTree = DTreeWrapper();
                nodes[idx].dTree.valid = false;
            }
            if (!nodes[idx].isLeaf()) {
                for (int i = 0; i < 2; i++) {
                    refine(nodes[idx]._children[i], maxSample, depth + 1);
                }
                AKR_CHECK(nodes[idx]._children[0] > 0 && nodes[idx]._children[1] > 0);
            }
        }

        void refine(size_t maxSample) {
            AKR_CHECK(maxSample > 0);
            for (auto &i : nodes) {
                if (i.isLeaf()) {
                    i.dTree.refine();
                }
            }
            refine(0, maxSample, 0);
            for (auto &i : nodes) {
                i.nSample = 0;
            }
        }
    };

    class GuidedPathTracer {
      public:
        const Scene *scene = nullptr;
        Sampler *sampler = nullptr;
        Spectrum L;
        Spectrum beta = Spectrum(1.0f);
        Allocator<> *allocator = nullptr;
        int depth = 0;
        int max_depth = 5;
        std::shared_ptr<STree> sTree;
        struct PPGVertex {
            vec3 wi;
            vec3 p;
            Spectrum L;
            Spectrum beta;
        };
        PPGVertex *vertices = nullptr;
        int n_vertices = 0;
        bool training = false;
        DTreeWrapper *dTree = nullptr;
        Float bsdfSamplingFraction = 0.5;
        static Float mis_weight(Float pdf_A, Float pdf_B) {
            pdf_A *= pdf_A;
            pdf_B *= pdf_B;
            return pdf_A / (pdf_A + pdf_B);
        }
        CameraSample camera_ray(const Camera *camera, const ivec2 &p) noexcept {
            CameraSample sample = camera->generate_ray(sampler->next2d(), sampler->next2d(), p);
            return sample;
        }
        astd::pair<const Light *, Float> select_light() noexcept { return scene->select_light(sampler->next2d()); }

        astd::optional<DirectLighting>
        compute_direct_lighting(SurfaceInteraction &si, const SurfaceHit &surface_hit,
                                const astd::pair<const Light *, Float> &selected) noexcept {
            auto [light, light_pdf] = selected;
            if (light) {
                DirectLighting lighting;
                LightSampleContext light_ctx;
                light_ctx.u = sampler->next2d();
                light_ctx.p = si.p;
                LightSample light_sample = light->sample_incidence(light_ctx);
                if (light_sample.pdf <= 0.0)
                    return astd::nullopt;
                light_pdf *= light_sample.pdf;
                auto f = light_sample.I * si.bsdf->evaluate(surface_hit.wo, light_sample.wi) *
                         std::abs(dot(si.ns, light_sample.wi));
                Float bsdf_pdf = si.bsdf->evaluate_pdf(surface_hit.wo, light_sample.wi) * bsdfSamplingFraction +
                                 (1.0 - bsdfSamplingFraction) * dTree->pdf(light_sample.wi);
                auto weight = mis_weight(light_pdf, bsdf_pdf);
                lighting.color = beta * f / light_pdf * mis_weight(light_pdf, bsdf_pdf) * weight;
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf = light_pdf;
                if (training)
                    sTree->deposit(si.p, light_sample.wi, luminance(Spectrum(weight * light_sample.I / light_pdf)));
                return lighting;
            } else {
                return astd::nullopt;
            }
        }

        void on_miss(const Ray &ray) noexcept {}

        void accumulate_radiance_wo_beta(const Spectrum &r) {
            L += beta * r;
            for (int i = 0; i < n_vertices && training; i++) {
                vertices[i].L += vertices[i].beta * r;
            }
        }

        void accumulate_beta(const Spectrum &k) {
            beta *= k;
            for (int i = 0; i < n_vertices && training; i++) {
                vertices[i].beta *= k;
            }
        }
        // @param mat_pdf: supplied if material is already chosen
        astd::optional<SurfaceVertex> on_surface_scatter(SurfaceInteraction &si, const SurfaceHit &surface_hit,
                                                        const astd::optional<PathVertex> &prev_vertex) noexcept {
            auto *material = surface_hit.material;
            auto wo = surface_hit.wo;
             MaterialEvalContext ctx = si.mat_eval_ctx(allocator, sampler);
            if (material->is_emissive()) {
                auto *emission = material->as_emissive();
                bool face_front = dot(wo, si.ng) > 0.0f;
                if (emission->double_sided || face_front) {
                    Spectrum I = emission->color->evaluate(ctx.sp);
                    if (depth == 0) {
                        accumulate_radiance_wo_beta(I);
                    } else {
                        vec3 prev_p = prev_vertex->p();
                        auto light = scene->get_light(surface_hit.geom_id, surface_hit.prim_id);
                        ReferencePoint ref;
                        ref.ng = prev_vertex->ng();
                        ref.p = prev_vertex->p();
                        auto light_pdf = light->pdf_incidence(ref, -wo) * scene->pdf_light(light);
                        Float weight_bsdf = mis_weight(prev_vertex->pdf(), light_pdf);
                        accumulate_radiance_wo_beta(weight_bsdf * I);
                    }
                    return astd::nullopt;
                }
            } else if (depth < max_depth) {
                auto u0 = sampler->next1d();
                auto u1 = sampler->next2d();
                auto u2 = sampler->next2d();
                SurfaceVertex vertex(si.triangle, surface_hit);
                si.bsdf = allocator->new_object<BSDF>(material->get_bsdf(ctx));
                BSDFSample sample;
                BSDFSampleContext sample_ctx(u1, wo);
                if (u0 < bsdfSamplingFraction) {
                    sample = si.bsdf->sample(sample_ctx);
                    sample.pdf *= bsdfSamplingFraction;
                    if (!(sample.sampled & BSDF_SPECULAR)) {
                        sample.pdf = sample.pdf + (1.0f - bsdfSamplingFraction) * dTree->pdf(sample.wi);
                    }
                } else {
                    auto w = dTree->sample(u1, u2);
                    sample.wi = w;
                    sample.pdf = dTree->pdf(w);
                    AKR_CHECK(sample.pdf >= 0);
                    sample.f = si.bsdf->evaluate(wo, sample.wi);
                    sample.sampled = static_cast<BSDFType>(BSDF_ALL & ~BSDF_SPECULAR);
                    sample.pdf *= 1.0f - bsdfSamplingFraction;
                    sample.pdf = sample.pdf + bsdfSamplingFraction * si.bsdf->evaluate_pdf(wo, sample.wi);
                }
                AKR_CHECK(!std::isnan(sample.pdf));
                AKR_CHECK(sample.pdf >= 0.0);
                AKR_CHECK(hmin(sample.f) >= 0.0f);
                if (std::isnan(sample.pdf) || sample.pdf == 0.0f) {
                    return astd::nullopt;
                }
                vertex.bsdf = si.bsdf;
                vertex.ray = Ray(si.p, sample.wi, Eps / std::abs(glm::dot(si.ng, sample.wi)));
                vertex.beta = sample.f * std::abs(glm::dot(si.ng, sample.wi)) / sample.pdf;
                vertex.pdf = sample.pdf;
                return vertex;
            }
            return astd::nullopt;
        }
        void run_megakernel(const Camera *camera, const ivec2 &p) noexcept {
            auto camera_sample = camera_ray(camera, p);
            Ray ray = camera_sample.ray;
            astd::optional<PathVertex> prev_vertex;
            while (true) {
                auto hit = scene->intersect(ray);
                if (!hit) {
                    on_miss(ray);
                    break;
                }
                SurfaceHit surface_hit(ray, *hit);
                auto trig = scene->get_triangle(surface_hit.geom_id, surface_hit.prim_id);
                surface_hit.material = trig.material;
                SurfaceInteraction si(surface_hit.uv, trig);
                dTree = sTree->dTree(si.p);
                auto vertex = on_surface_scatter(si, surface_hit, prev_vertex);
                if (!vertex) {
                    break;
                }
                astd::optional<DirectLighting> has_direct = compute_direct_lighting(si, surface_hit, select_light());
                if (has_direct) {
                    auto &direct = *has_direct;
                    if (!is_black(direct.color) && !scene->occlude(direct.shadow_ray)) {
                        accumulate_radiance_wo_beta(direct.color);
                    }
                }
                vertices[n_vertices].p = si.p;
                vertices[n_vertices].wi = vertex->ray.d;
                vertices[n_vertices].beta = Spectrum(1.0 / vertex->pdf);
                accumulate_beta(vertex->beta);
                n_vertices++;
                depth++;
                ray = vertex->ray;
                prev_vertex = PathVertex(*vertex);
            }
            if (training) {
                for (int i = 0; i < n_vertices; i++) {
                    auto irradiance = luminance(clamp_zero(vertices[i].L));
                    AKR_CHECK(irradiance >= 0);
                    sTree->deposit(vertices[i].p, vertices[i].wi, irradiance);
                }
            }
        }
    };

    class GuidedPathTracerIntegrator : public Integrator {
        int spp;
        int max_depth;
        const int tile_size = 16;
        std::shared_ptr<STree> sTree;
        int trainingSamples = 16;

      public:
        GuidedPathTracerIntegrator(int spp, int max_depth, int trainingSamples)
            : spp(spp), max_depth(max_depth), trainingSamples(trainingSamples) {}
        void render(const Scene *scene, Film *film) override {
            sTree.reset(new STree(scene->accel->world_bounds()));
            
            AKR_ASSERT_THROW(glm::all(glm::equal(film->resolution(), scene->camera->resolution())));
            auto n_tiles = ivec2(film->resolution() + ivec2(tile_size - 1)) / ivec2(tile_size);
            debug("resolution: {}, tile size: {}, tiles: {}", film->resolution(), tile_size, n_tiles);
            std::mutex mutex;
            std::vector<astd::pmr::monotonic_buffer_resource *> resources;
            for (size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
                resources.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::get_default_resource()));
            }
            uint32_t pass = 0;
            uint32_t accumulatedSamples = 0;
            for (pass = 0; accumulatedSamples < (uint32_t)trainingSamples; pass++) {
                size_t samples;
                samples = 1ull << pass;
                auto nextPassSamples = 2u << pass;
                if (accumulatedSamples + samples + nextPassSamples > (uint32_t)trainingSamples) {
                    samples = (uint32_t)trainingSamples - accumulatedSamples;
                }

                info("Learning pass {}, spp:{}", pass + 1, samples);
                accumulatedSamples += samples;
                parallel_for_2d(n_tiles, [=, &resources](const ivec2 &tile_pos, int tid) {
                    Allocator<> allocator(resources[tid]);
                    Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + ivec2(1)) * (int)tile_size};
                    auto tile = film->tile(tileBounds);
                    auto camera = scene->camera;
                    auto sampler = scene->sampler->clone(&allocator);
                    for (int y = tile.bounds.pmin.y; y < tile.bounds.pmax.y; y++) {
                        for (int x = tile.bounds.pmin.x; x < tile.bounds.pmax.x; x++) {
                            sampler->set_sample_index(x + y * film->resolution().x);
                            for (int s = 0; s < samples; s++) {
                                sampler->start_next_sample();
                                GuidedPathTracer pt;
                                pt.sTree = sTree;
                                pt.n_vertices = 0;
                                pt.vertices = allocator.allocate_object<GuidedPathTracer::PPGVertex>(max_depth + 1);
                                pt.training = true;
                                pt.scene = scene;
                                pt.allocator = &allocator;
                                pt.sampler = sampler;
                                pt.L = Spectrum(0.0);
                                pt.beta = Spectrum(1.0);
                                pt.max_depth = max_depth;
                                pt.run_megakernel(camera, ivec2(x, y));
                            }
                        }
                    }
                    resources[tid]->release();
                });
                info("Refining SDTree; pass: {}", pass + 1);
                info("nodes: {}", sTree->nodes.size());
                sTree->refine(12000 * std::sqrt(samples));
            }

            parallel_for_2d(n_tiles, [=, &mutex, &resources](const ivec2 &tile_pos, int tid) {
                Allocator<> allocator(resources[tid]);
                Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + ivec2(1)) * (int)tile_size};
                auto tile = film->tile(tileBounds);
                auto camera = scene->camera;
                auto sampler = scene->sampler->clone(&allocator);
                for (int y = tile.bounds.pmin.y; y < tile.bounds.pmax.y; y++) {
                    for (int x = tile.bounds.pmin.x; x < tile.bounds.pmax.x; x++) {
                        sampler->set_sample_index(x + y * film->resolution().x);
                        for (int s = 0; s < spp; s++) {
                            sampler->start_next_sample();
                            GuidedPathTracer pt;
                            pt.scene = scene;
                            pt.sTree = sTree;
                            pt.n_vertices = 0;
                            pt.vertices = allocator.allocate_object<GuidedPathTracer::PPGVertex>(max_depth + 1);
                            pt.training = false;
                            pt.allocator = &allocator;
                            pt.sampler = sampler;
                            pt.L = Spectrum(0.0);
                            pt.beta = Spectrum(1.0);
                            pt.max_depth = max_depth;
                            pt.run_megakernel(camera, ivec2(x, y));
                            tile.add_sample(vec2(x, y), pt.L, 1.0f);
                        }
                    }
                }
                std::lock_guard<std::mutex> _(mutex);
                film->merge_tile(tile);
                resources[tid]->release();
            });
            for (auto rsrc : resources) {
                delete rsrc;
            }
        }
    };
    class GuidedPathIntegratorNode final : public IntegratorNode {
      public:
        int spp = 16;
        int max_depth = 5;
        int training_samples = 16;
        Integrator *create_integrator(Allocator<> *allocator) override {
            return allocator->new_object<GuidedPathTracerIntegrator>(spp, max_depth, training_samples);
        }
        const char *description() override { return "[Path Tracer]"; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "spp") {
                spp = value.get<int>().value();
            } else if (field == "max_depth") {
                max_depth = value.get<int>().value();
            } else if (field == "training_samples") {
                training_samples = value.get<int>().value();
            }
        }
        int get_spp() const override { return spp; }
        bool set_spp(int spp) override { return false; }
    };
    AKR_EXPORT_NODE(GuidedPath, GuidedPathIntegratorNode)
} // namespace akari::render