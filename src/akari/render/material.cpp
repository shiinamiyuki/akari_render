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
#include <akari/render/scenegraph.h>
#include <akari/render/material.h>
namespace akari::render {
    NullBSDF *null_bsdf() {
        static NullBSDF bsdf;
        return &bsdf;
    }
        class MixMaterialNode final : public MaterialNode {
        std::shared_ptr<TextureNode> fraction;
        std::shared_ptr<MaterialNode> mat_A;
        std::shared_ptr<MaterialNode> mat_B;

      public:
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "fraction" || field == "frac") {
                fraction = resolve_texture(value);
            } else if (field == "first") {
                mat_A = dyn_cast<MaterialNode>(value.object());
                AKR_ASSERT_THROW(mat_A);
            } else if (field == "second") {
                mat_B = dyn_cast<MaterialNode>(value.object());
                AKR_ASSERT_THROW(mat_B);
            }
        }
        Material *create_material(Allocator<> *allocator) override {
            return allocator->new_object<MixMaterial>(fraction->create_texture(allocator),
                                                      mat_A->create_material(allocator),
                                                      mat_B->create_material(allocator));
        }
    };

    std::shared_ptr<MaterialNode> make_mix_material() { return std::make_shared<MixMaterialNode>(); }
} // namespace akari::render