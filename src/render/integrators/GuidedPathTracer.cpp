// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#include <akari/Core/Logger.h>
#include <akari/Core/Parallel.h>
#include <akari/Core/Plugin.h>
#include <akari/Render/Integrator.h>

#include <akari/Core/Progress.hpp>
#include <future>
#include <mutex>
#include <stack>
namespace akari {

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
                AKARI_CHECK(_sum[idx].value() >= 0.0f);
                return 4.0f * float(_sum[idx]);
            }
        }

        float pdf(const vec2 &p, std::vector<QTreeNode> &nodes) const {
            auto idx = childIndex(p);
            //            if (!(_sum[idx].value() > 0)) {
            //                return 0.0f;
            //            }
            auto s = sum();
            AKARI_CHECK(s >= 0);
            AKARI_CHECK(!std::isnan(s));
            auto factor = s <= 0.0f ? 0.25f : _sum[idx].value() / s;
            if (factor < 0) {
                Info("{} {} {}\n", factor, _sum[idx].value(), s);
            }
            AKARI_CHECK(factor >= 0);
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
            //            AKARI_CHECK(total > 0);
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
            AKARI_CHECK(e >= 0);
            AKARI_CHECK(_sum[idx].value() >= 0);
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
                size_t node = -1, otherNode = -1;
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
                    AKARI_CHECK(node.otherNode >= 0);
                    auto &otherNode = node.tree->nodes.at(node.otherNode);
                    auto fraction = total == 0.0f ? std::pow(0.25, node.depth) : otherNode._sum[i].value() / total;
                    // log::log("{} {}\n", otherNode._sum[i].value(), fraction);
                    if (fraction >= threshold && node.depth < 10) {
                        if (otherNode.isLeaf(i)) {
                            stack.push({nodes.size(), nodes.size(), this, node.depth + 1});
                        } else {
                            AKARI_CHECK(otherNode._children[i] > 0);
                            AKARI_CHECK((size_t)otherNode._children[i] != node.otherNode);
                            AKARI_CHECK(node.tree == &prev);
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
            AKARI_CHECK(!building.nodes.empty());
            auto p = dirToCanonical(w);
            building.deposit(p, e);
        }

        void refine() {
            AKARI_CHECK(building.sum.value() >= 0.0f);
            //            building._build();
            sampling = building;
            sampling._build();
            AKARI_CHECK(sampling.sum.value() >= 0.0f);
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
            //            AKARI_CHECK(0.0f - 1e-6f <= p[axis] && p[axis] <= 1.0f + 1e-6f);
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
            auto sz = MaxComp(box.Size()) * 0.5f;
            auto centroid = box.Centroid();
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
                    AKARI_CHECK(node.isLeaf());
                }
                nodes[idx]._isLeaf = false;
                nodes[idx].dTree = DTreeWrapper();
                nodes[idx].dTree.valid = false;
            }
            if (!nodes[idx].isLeaf()) {
                for (int i = 0; i < 2; i++) {
                    refine(nodes[idx]._children[i], maxSample, depth + 1);
                }
                AKARI_CHECK(nodes[idx]._children[0] > 0 && nodes[idx]._children[1] > 0);
            }
        }

        void refine(size_t maxSample) {
            AKARI_CHECK(maxSample > 0);
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

    static Float MisWeight(Float pdfA, Float pdfB) {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }
    struct GPTRenderTask : RenderTask {
        RenderContext ctx;
        std::mutex mutex;
        std::condition_variable filmAvailable, done;
        std::future<void> future;
        int spp;
        int minDepth;
        int maxDepth;
        int trainingSamples = 16;
        bool enableRR;
        std::shared_ptr<STree> sTree;
        GPTRenderTask(const RenderContext &ctx, int spp, int minDepth, int maxDepth, int trainingSamples, bool enableRR)
            : ctx(ctx), spp(spp), minDepth(minDepth), maxDepth(maxDepth), trainingSamples(trainingSamples),
              enableRR(enableRR) {
            sTree.reset(new STree(ctx.scene->GetBounds()));
        }
        bool HasFilmUpdate() override { return false; }
        std::shared_ptr<const Film> GetFilmUpdate() override { return ctx.camera->GetFilm(); }
        bool IsDone() override { return false; }
        bool WaitEvent(Event event) override {
            if (event == Event::EFILM_AVAILABLE) {
                std::unique_lock<std::mutex> lock(mutex);

                filmAvailable.wait(lock, [=]() { return HasFilmUpdate() || IsDone(); });
                return HasFilmUpdate();
            } else if (event == Event::ERENDER_DONE) {
                std::unique_lock<std::mutex> lock(mutex);

                done.wait(lock, [=]() { return IsDone(); });
                return true;
            }
            return false;
        }
        void Wait() override {
            future.wait();
            done.notify_all();
        }
        Spectrum BackgroundLi(const Ray &ray) { return Spectrum(0); }
        static std::string PrintVec3(const vec3 &v) { return fmt::format("{} {} {}", v.x, v.y, v.z); }

        struct PathVertex {
            vec3 wi;
            vec3 p;
            Spectrum L;
            Spectrum beta;
        };

        Spectrum Li(bool enableNEE, bool training, Ray ray, Sampler *sampler, MemoryArena &arena) {
            auto scene = ctx.scene;
            auto vertices = arena.allocN<PathVertex>(maxDepth + 1);
            Spectrum Li(0), beta(1);
            bool specular = false;
            Float prevScatteringPdf = 0;
            Interaction *prevInteraction = nullptr;
            int nVertices = 0;
            auto addRadiance = [&](const Spectrum &L) {
                for (int i = 0; i < nVertices && training; i++) {
                    vertices[i].L += vertices[i].beta * L;
                }
                Li += beta * L;
            };
            auto updateBeta = [&](const Spectrum &k) {
                beta *= k;
                for (int i = 0; i < nVertices && training; i++) {
                    vertices[i].beta *= k;
                }
            };
            for (int i = 0; i < maxDepth + 1; i++) {
                vertices[i].L = Spectrum(0);
                vertices[i].beta = Spectrum(1);
            }
            const auto bsdfSamplingFraction = 0.5f;
            int depth = 0;
            while (true) {
                Intersection intersection(ray);
                if (scene->intersect(ray, &intersection)) {
                    auto &mesh = scene->get_mesh(intersection.meshId);
                    int group = mesh.get_primitive_group(intersection.primId);
                    const auto &materialSlot = mesh.get_material_slot(group);
                    const auto *light = mesh.get_light(intersection.primId);

                    auto material = materialSlot.material;
                    if (!material) {
                        Debug("no material!!\n");
                        break;
                    }
                    Triangle triangle{};
                    mesh.get_triangle(intersection.primId, &triangle);
                    const auto &p = intersection.p;
                    auto *si = arena.alloc<SurfaceInteraction>(&materialSlot, -ray.d, p, triangle, intersection, arena);
                    si->compute_scattering_functions(arena, TransportMode::EImportance, 1.0f);
                    auto dTree = sTree->dTree(si->p);
                    auto Le = si->Le(si->wo);
                    if (!Le.is_black()) {
                        if (!light || !enableNEE || specular || depth == 0)
                            addRadiance(light->Li(si->wo, si->uv));
                        else {
                            auto lightPdf = light->pdf_incidence(*prevInteraction, ray.d) * scene->PdfLight(light);
                            addRadiance(light->Li(si->wo, si->uv) * MisWeight(prevScatteringPdf, lightPdf));
                        }
                    }
                    if (depth++ >= maxDepth) {
                        break;
                    }
                    auto u0 = sampler->next1d();
                    auto u1 = sampler->next2d();
                    auto u2 = sampler->next2d();
                    BSDFSample bsdfSample(u1.x, u2, *si);
                    // BSDF Sampling
                    {
                        //                    log::log("{}\n", reinterpret_cast<size_t>(dTree));
                        if (u0 < bsdfSamplingFraction) {
                            si->bsdf->sample(bsdfSample);
                            AKARI_CHECK(bsdfSample.pdf >= 0);
                            bsdfSample.pdf *= bsdfSamplingFraction;
                            if (!(bsdfSample.sampledType & BSDF_SPECULAR)) {

                                bsdfSample.pdf =
                                    bsdfSample.pdf + (1.0f - bsdfSamplingFraction) * dTree->pdf(bsdfSample.wi);
                            }
                        } else {
                            auto w = dTree->sample(u1, u2);
                            bsdfSample.wi = w;
                            bsdfSample.pdf = dTree->pdf(w);
                            AKARI_CHECK(bsdfSample.pdf >= 0);
                            bsdfSample.f = si->bsdf->evaluate(bsdfSample.wo, bsdfSample.wi);
                            bsdfSample.sampledType = static_cast<BSDFType>(BSDF_ALL & ~BSDF_SPECULAR);
                            bsdfSample.pdf *= 1.0f - bsdfSamplingFraction;
                            bsdfSample.pdf = bsdfSample.pdf +
                                             bsdfSamplingFraction * si->bsdf->evaluate_pdf(bsdfSample.wo, bsdfSample.wi);
                        }

                        AKARI_CHECK(!std::isnan(bsdfSample.pdf));
                        AKARI_CHECK(bsdfSample.pdf >= 0.0);
                        AKARI_CHECK(MinComp(bsdfSample.f) >= 0.0f);
                        if (std::isnan(bsdfSample.pdf) || bsdfSample.pdf <= 0.0f) {
                            break;
                        }
                    }

                    assert(bsdfSample.pdf >= 0);
                    if (bsdfSample.pdf <= 0) {
                        break;
                    }
                    specular = bsdfSample.sampledType & BSDF_SPECULAR;
                    if (enableNEE) {
                        Float lightPdf = 0;
                        auto sampledLight = scene->sample_one_light(sampler->next1d(), &lightPdf);
                        if (sampledLight && lightPdf > 0) {
                            LightSample lightSample{};
                            VisibilityTester tester{};
                            sampledLight->sample_incidence(sampler->next2d(), *si, &lightSample, &tester);
                            lightPdf *= lightSample.pdf;
                            auto wi = lightSample.wi;
                            auto wo = si->wo;
                            auto absCos = abs(dot(lightSample.wi, si->Ns));
                            auto f = si->bsdf->evaluate(wo, wi) * absCos;
                            Spectrum radiance;
                            if (lightPdf > 0 && MaxComp(f) > 0 && tester.visible(*scene)) {
                                Float weight = 1;
                                if (specular) {
                                    radiance = (f * lightSample.I / lightPdf);
                                } else {
                                    auto scatteringPdf = si->bsdf->evaluate_pdf(wo, wi);
                                    weight = MisWeight(lightPdf, scatteringPdf);
                                    radiance = (f * lightSample.I / lightPdf * weight);
                                }
                                if (training && !enableNEE) {
                                    sTree->deposit(intersection.p, lightSample.wi,
                                                   Spectrum(weight * lightSample.I / lightPdf / absCos).luminance());
                                }
                                addRadiance(radiance);
                            }
                        }
                    }
                    prevScatteringPdf = bsdfSample.pdf;
                    auto wiW = bsdfSample.wi;
                    vertices[nVertices].p = intersection.p;
                    vertices[nVertices].wi = wiW;
                    vertices[nVertices].beta = Spectrum(1 / bsdfSample.pdf);
                    updateBeta(bsdfSample.f * abs(dot(wiW, si->Ns)) / bsdfSample.pdf);

                    if (enableRR) {
                        if (depth > minDepth) {
                            Float continueProb = std::min(0.95f, MaxComp(beta));
                            if (sampler->next1d() < continueProb) {
                                updateBeta(Spectrum(1.0f) / continueProb);
                            } else {
                                break;
                            }
                        }
                    }
                    nVertices++;
                    ray = si->spawn_dir(wiW);
                    prevInteraction = si;
                } else {
                    addRadiance(BackgroundLi(ray));
                    break;
                }
            }
            if (training) {
                for (int i = 0; i < nVertices; i++) {
                    auto irradiance = vertices[i].L.remove_nans().luminance();
                    AKARI_CHECK(irradiance >= 0);
                    sTree->deposit(vertices[i].p, vertices[i].wi, irradiance);
                }
            }
            return Li;
        }

        void Train() {
            auto scene = ctx.scene;
            auto &camera = ctx.camera;
            auto &_sampler = ctx.sampler;
            auto film = camera->GetFilm();
            auto nTiles = ivec2(film->Dimension() + ivec2(TileSize - 1)) / ivec2(TileSize);
            uint32_t pass = 0;
            uint32_t accumulatedSamples = 0;
            for (pass = 0; accumulatedSamples < (uint32_t)trainingSamples; pass++) {
                size_t samples;
                samples = 1u << pass; // 2 * std::pow(1.1, pass);//1ull << pass;
                auto nextPassSamples = 2u << pass;
                if (accumulatedSamples + samples + nextPassSamples > (uint32_t)trainingSamples) {
                    samples = (uint32_t)trainingSamples - accumulatedSamples;
                }

                Info("Learning pass {}, spp:{}\n", pass + 1, samples);
                accumulatedSamples += samples;
                parallel_for_2d(nTiles, [=](ivec2 tilePos, uint32_t tid) {
                    (void) tid;
                    MemoryArena arena;
                    Bounds2i tileBounds = Bounds2i{tilePos * (int) TileSize, (tilePos + ivec2(1)) * (int) TileSize};
                    auto tile = film->GetTile(tileBounds);
                    auto sampler = _sampler->clone();
                    for (int y = tile.bounds.p_min.y; y < tile.bounds.p_max.y; y++) {
                        for (int x = tile.bounds.p_min.x; x < tile.bounds.p_max.x; x++) {
                            sampler->set_sample_index(x + y * film->Dimension().x);
                            for (uint32_t s = 0; s < samples; s++) {
                                //                          sampler->startSample(accumulatedSamples);
                                sampler->start_next_sample();
                                CameraSample sample;
                                camera->generate_ray(sampler->next2d(), sampler->next2d(), ivec2(x, y), &sample);
                                auto Li = this->Li(true, true, sample.primary, sampler.get(), arena);
                                (void) Li;
                                arena.reset();
                            }
                        }
                    }
                });
                Info("Refining SDTree; pass: {}\n", pass + 1);
                Info("nodes: {}\n", sTree->nodes.size());
                sTree->refine(12000 * std::sqrt(samples));
            }
        }
        void Start() override {
            future = std::async(std::launch::async, [=]() {
                auto beginTime = std::chrono::high_resolution_clock::now();
                Train();
                auto scene = ctx.scene;
                auto &camera = ctx.camera;
                auto &_sampler = ctx.sampler;
                auto film = camera->GetFilm();
                auto nTiles = ivec2(film->Dimension() + ivec2(TileSize - 1)) / ivec2(TileSize);
                ProgressReporter progressReporter(nTiles.x * nTiles.y, [=](size_t cur, size_t tot) {
                    if (spp <= 16) {
                        if (cur % (tot / 10) == 0) {
                            show_progress(double(cur) / tot, 70);
                        }
                    } else {
                        show_progress(double(cur) / tot, 70);
                    }
                });
                parallel_for_2d(nTiles, [=, &progressReporter](ivec2 tilePos, uint32_t tid) {
                    (void) tid;
                    MemoryArena arena;
                    Bounds2i tileBounds = Bounds2i{tilePos * (int) TileSize, (tilePos + ivec2(1)) * (int) TileSize};
                    auto tile = film->GetTile(tileBounds);
                    auto sampler = _sampler->clone();
                    for (int y = tile.bounds.p_min.y; y < tile.bounds.p_max.y; y++) {
                        for (int x = tile.bounds.p_min.x; x < tile.bounds.p_max.x; x++) {
                            sampler->set_sample_index(x + y * film->Dimension().x);
                            for (int s = 0; s < spp; s++) {
                                sampler->start_next_sample();
                                CameraSample sample;
                                camera->generate_ray(sampler->next2d(), sampler->next2d(), ivec2(x, y), &sample);
                                auto Li = this->Li(true, false, sample.primary, sampler.get(), arena);
                                arena.reset();
                                tile.AddSample(ivec2(x, y), Li, 1.0f);
                            }
                        }
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    film->merge_tile(tile);
                    progressReporter.update();
                });
                auto endTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = (endTime - beginTime);
                Info("Rendering done in {} secs, traced {} rays, {} M rays/sec\n", elapsed.count(),
                     scene->GetRayCounter(), scene->GetRayCounter() / elapsed.count() / 1e6);
            });
        }
    };

    class GuidedPathTracer : public Integrator {
        int spp = 16;
        int minDepth = 5, maxDepth = 16;
        int trainingSamples = 16;
        bool enableRR = false;

      public:
        AKR_DECL_COMP(GuidedPathTracer, "GuidedPathTracer")
        AKR_SER(spp, trainingSamples, minDepth, maxDepth, enableRR)
        std::shared_ptr<RenderTask> create_render_task(const RenderContext &ctx) override {
            return std::make_shared<GPTRenderTask>(ctx, spp, minDepth, maxDepth, trainingSamples, enableRR);
        }
    };
    AKR_EXPORT_COMP(GuidedPathTracer, "Integrator");
} // namespace akari