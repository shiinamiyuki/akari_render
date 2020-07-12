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
#include <fstream>
#include <akari/core/plugin.h>
#include <akari/core/spectrum.h>
#include <akari/render/texture.h>
#include <akari/asl/asl.h>
namespace akari {
    class ASLTexture final : public Texture {
      public:
        [[refl]] fs::path path;
        std::shared_ptr<asl::Program> program;
        vec3 (*fp)(ShadingPoint) = nullptr;
        ASLTexture()=default;
        ASLTexture(const fs::path & path):path(path){}
        AKR_IMPLS(Texture)
        Spectrum evaluate(const ShadingPoint &sp) const override { 
            auto s = fp(sp);
            return Spectrum(s.x,s.y,s.z);
        }
        Float average_luminance() const override { return 1; }
        void commit() {
            std::vector<asl::TranslationUnit> units;
            std::ifstream src_file(path);
            std::string src((std::istreambuf_iterator<char>(src_file)),
                 std::istreambuf_iterator<char>());
            units.emplace_back(asl::TranslationUnit{path.string(), src});
            program = asl::compile(units).extract_value();
            fp = reinterpret_cast<vec3(*)(ShadingPoint)>(program->get_function_pointer("main"));
        }
    };
    AKR_EXPORT std::shared_ptr<Texture> create_asl_texture(const fs::path& path){
        return std::make_shared<ASLTexture>(path);
    }
#include "generated/ASLTexture.hpp"
    AKR_EXPORT_PLUGIN(p) {

    }
}