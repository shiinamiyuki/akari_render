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
#include <akari/core/logger.h>
namespace akari {
    static void _dummy(vec3*s,ShadingPoint* ){*s = vec3(0);}
    class ASLTexture final : public Texture {
      public:
        [[refl]] fs::path path;
        std::shared_ptr<asl::Program> program;
        void (*fp)(vec3 *, ShadingPoint*) = nullptr;
        ASLTexture()=default;
        ASLTexture(const fs::path & path):path(path){}
        AKR_IMPLS(Texture)
        Spectrum evaluate(const ShadingPoint &sp) const override { 
            ShadingPoint _ = sp;
            vec3 s;
            fp(&s, &_);
            s = max(vec3(0), s);
            return Spectrum(s.x,s.y,s.z);
        }
        void commit() {
            std::vector<asl::TranslationUnit> units;
            std::ifstream src_file(path);
            std::string src((std::istreambuf_iterator<char>(src_file)),
                 std::istreambuf_iterator<char>());
            units.emplace_back(asl::TranslationUnit{path.string(), src});
            asl::CompileOptions opt;
            // opt.opt_level = asl::CompileOptions::OFF;
            auto result = asl::compile(units, opt);
            if(result.has_value()){
                program = result.extract_value();
                fp = reinterpret_cast<decltype(fp)>(program->get_function_pointer("main"));
            }else{
                error("error when compile {}:\n{}", path.string(), result.extract_error().what());
                fp = &_dummy;
            }
        }
    };
    AKR_EXPORT std::shared_ptr<Texture> create_asl_texture(const fs::path& path){
        return std::make_shared<ASLTexture>(path);
    }
#include "generated/ASLTexture.hpp"
    AKR_EXPORT_PLUGIN(p) {

    }
}