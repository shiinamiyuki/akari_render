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
#include <akari/core/parallel.h>
#include <akari/core/progress.hpp>
#include <akari/core/profiler.h>
#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/common.h>
#include <akari/render/pathtracer.h>

namespace akari::render {
    struct ManifoldVertex {
        Vec3 p;
        Vec3 dpdu, dpdv;
        Vec3 n;
        Vec3 dndu, dndv;
        Float eta;
        Matrix2f A, B, C;
    };
    using ManifoldPath = std::vector<ManifoldVertex>;

    // https://www.cs.cornell.edu/projects/manifolds-sg12/manifolds-sg12-tr.pdf
    void compute_derivatives(const ManifoldPath &path, ManifoldVertex *v) {
        Vec3 wi = v[-1].p - v[0].p;
        Vec3 wo = v[1].p - v[0].p;
        float ili = 1 / wi.length();
        float ilo = 1 / wo.length();
        wi *= ili;
        wo *= ilo;
        Vec3 H = wi + v[0].eta * wo;
        float ilh = 1 / H.length();
        H *= ilh;
        float dot_H_n = dot(v[0].n, H), dot_H_dndu = dot(v[0].dndu, H), dot_H_dndv = dot(v[0].dndv, H),
              dot_u_n = dot(v[0].dpdu, v[0].n), dot_v_n = dot(v[0].dpdv, v[0].n);
        /* Local shading tangent frame */
        Vec3 s = v[0].dpdu - dot_u_n * v[0].n;
        Vec3 t = v[0].dpdv - dot_v_n * v[0].n;
        ilo *= v[0].eta * ilh;
        ili *= ilh;
        /* Derivatives of C with respect to x_{i-1} */
        Vec3 dH_du = (v[-1].dpdu - wi * dot(wi, v[-1].dpdu)) * ili,
             dH_dv = (v[-1].dpdv - wi * dot(wi, v[-1].dpdv)) * ili;
        dH_du -= H * dot(dH_du, H);
        dH_dv -= H * dot(dH_dv, H);
        v[0].A = Matrix2f(dot(dH_du, s), dot(dH_dv, s), dot(dH_du, t), dot(dH_dv, t));
        /* Derivatives of C with respect to x_i */
        dH_du = -v[0].dpdu * (ili + ilo) + wi * (dot(wi, v[0].dpdu) * ili) + wo * (dot(wo, v[0].dpdu) * ilo);
        dH_dv = -v[0].dpdv * (ili + ilo) + wi * (dot(wi, v[0].dpdv) * ili) + wo * (dot(wo, v[0].dpdv) * ilo);
        dH_du -= H * dot(dH_du, H);
        dH_dv -= H * dot(dH_dv, H);
        v[0].B = Matrix2f(dot(dH_du, s) - dot(v[0].dpdu, v[0].dndu) * dot_H_n - dot_u_n * dot_H_dndu,
                          dot(dH_dv, s) - dot(v[0].dpdu, v[0].dndv) * dot_H_n - dot_u_n * dot_H_dndv,
                          dot(dH_du, t) - dot(v[0].dpdv, v[0].dndu) * dot_H_n - dot_v_n * dot_H_dndu,
                          dot(dH_dv, t) - dot(v[0].dpdv, v[0].dndv) * dot_H_n - dot_v_n * dot_H_dndv);
        /* Derivatives of C with respect to x_{i+1} */
        dH_du = (v[1].dpdu - wo * dot(wo, v[1].dpdu)) * ilo;
        dH_dv = (v[1].dpdv - wo * dot(wo, v[1].dpdv)) * ilo;
        dH_du -= H * dot(dH_du, H);
        dH_dv -= H * dot(dH_dv, H);
        v[0].C = Matrix2f(dot(dH_du, s), dot(dH_dv, s), dot(dH_du, t), dot(dH_dv, t));
    }
    std::optional<Vec3> compute_step(const ManifoldVertex & v0, const Vec3 &n_offset){
        
    }
    class ManifoldPathIntegrator : public UniAOVIntegrator {
        const int tile_size = 16;
        int spp = 16;

      public:
    };
} // namespace akari::render