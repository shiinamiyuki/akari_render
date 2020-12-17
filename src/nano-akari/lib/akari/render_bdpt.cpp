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
    namespace bidir {
        struct SurfaceVertex {
            SurfaceInteraction si;
            Vec3 wo;
            Ray ray;
            Spectrum beta;
            std::optional<BSDF> bsdf;
            Float pdf_fwd = 0.0;
            Float pdf_rev = 0.0;
            BSDFType sampled_lobe = BSDFType::Unset;
            // SurfaceVertex() = default;
            SurfaceVertex(const Vec3 &wo, const SurfaceInteraction si) : wo(wo), si(si) {}
            Vec3 p() const { return si.p; }
            Vec3 ng() const { return si.ng; }
        };
        struct CameraVertex {};
        struct LightVertex {};

        struct Vertex : Variant<CameraVertex, LightVertex, SurfaceVertex> {
            using Variant::Variant;
        };

        using Path = BufferView<Vertex>;
    } // namespace bidir
    
    Film render_bdpt(PTConfig config, const Scene &scene) {
        Film film(scene.camera->resolution());
        thread::parallel_for(thread::blocked_range<2>(film.resolution(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            //  film.add_sample(id, Spectrum(1.0), 1.0);
        });
        spdlog::info("render bdpt done");
        return film;
    }
} // namespace akari::render