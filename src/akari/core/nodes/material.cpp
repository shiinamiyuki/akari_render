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
#include <akari/core/nodes/material.h>
#include <akari/kernel/material.h>
#include <akari/core/logger.h>
#include <akari/core/resource.h>
namespace akari {
    AKR_VARIANT class TextureNode : public SceneGraphNode<C> {
      public:
        AKR_IMPORT_TYPES()
        virtual Texture<C> *compile(MemoryArena *arena) = 0;
    };

    AKR_VARIANT class ConstantTextureNode : public TextureNode<C> {
      public:
        Color3f color;
        AKR_IMPORT_TYPES()
        ConstantTextureNode() = default;
        ConstantTextureNode(Color3f color) : color(color) {}
        Texture<C> *compile(MemoryArena *arena) override { return arena->alloc<Texture<C>>(ConstantTexture<C>(color)); }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {}
    };

    AKR_VARIANT class ImageTextureNode : public TextureNode<C> {
      public:
        std::shared_ptr<RGBAImage> image;
        AKR_IMPORT_TYPES()
        ImageTextureNode() = default;
        ImageTextureNode(const fs::path &path) {
            auto res = resource_manager()->load_path<ImageResource>(path);
            if (res) {
                image = res.extract_value()->image();
            } else {
                auto err = res.extract_error();
                error("error loading {}: {}", path.string(), err.what());
                throw std::runtime_error("Error loading image");
            }
        }
        Texture<C> *compile(MemoryArena *arena) override {
            return arena->alloc<Texture<C>>(ImageTexture<C>(image.get()));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {}
    };
    AKR_VARIANT
    static std::shared_ptr<TextureNode<C>> resolve_texture(const sdl::Value &value) {
        AKR_IMPORT_TYPES()
        if (value.is_array()) {
            return std::make_shared<ConstantTextureNode<C>>(load_array<Color3f>(value));
        }else if (value.is_number()) {
            return std::make_shared<ConstantTextureNode<C>>(Color3f(value.get<float>().value()));
        } else if (value.is_string()) {
            auto path = value.get<std::string>().value();
            return std::make_shared<ImageTextureNode<C>>(path);
        } else {
            AKR_ASSERT_THROW(value.is_object());
            auto tex = dyn_cast<TextureNode<C>>(value.object());
            AKR_ASSERT_THROW(tex);
            return tex;
        }
    }
    AKR_VARIANT class DiffuseMaterialNode : public MaterialNode<C> {
      public:
        AKR_IMPORT_TYPES()
        std::shared_ptr<TextureNode<C>> color;
        Material<C> *compile(MemoryArena *arena) override {
            auto tex = color->compile(arena);
            return arena->alloc<Material<C>>(DiffuseMaterial<C>(tex));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = resolve_texture<C>(value);
            }
        }
    };
    AKR_VARIANT class GlossyMaterialNode : public MaterialNode<C> {
      public:
        AKR_IMPORT_TYPES()
        std::shared_ptr<TextureNode<C>> color;
        std::shared_ptr<TextureNode<C>> roughness;
        Material<C> *compile(MemoryArena *arena) override {
            auto tex = color->compile(arena);
            auto tex_rough = roughness->compile(arena);
            return arena->alloc<Material<C>>(GlossyMaterial<C>(tex, tex_rough));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = resolve_texture<C>(value);
            } else if (field == "roughness") {
                roughness = resolve_texture<C>(value);
            }
        }
    };
    AKR_VARIANT class MixMaterialNode : public MaterialNode<C> {
      public:
        AKR_IMPORT_TYPES()
        std::shared_ptr<TextureNode<C>> fraction;
        std::shared_ptr<MaterialNode<C>> first, second;
        Material<C> *compile(MemoryArena *arena) override {
            auto tex = fraction->compile(arena);
            auto a = first->compile(arena);
            auto b = second->compile(arena);
            return arena->alloc<Material<C>>(MixMaterial<C>(tex, a, b));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "first") {
                first = dyn_cast<MaterialNode<C>>(value.object());
                AKR_ASSERT_THROW(first);
            } else if (field == "second") {
                second = dyn_cast<MaterialNode<C>>(value.object());
                AKR_ASSERT_THROW(first);
            } else if (field == "fraction") {
                fraction = resolve_texture<C>(value);
            }
        }
    };
    AKR_VARIANT class EmissiveMaterialNode : public MaterialNode<C> {
      public:
        AKR_IMPORT_TYPES()
        bool double_sided = false;
        std::shared_ptr<TextureNode<C>> color;
        Material<C> *compile(MemoryArena *arena) override {
            auto tex = color->compile(arena);
            return arena->alloc<Material<C>>(EmissiveMaterial<C>(tex));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = resolve_texture<C>(value);
            }
        }
    };

    AKR_VARIANT void RegisterMaterialNode<C>::register_nodes() {
        AKR_IMPORT_TYPES()
        register_node<C, DiffuseMaterialNode<C>>("DiffuseMaterial");
        register_node<C, EmissiveMaterialNode<C>>("EmissiveMaterial");
        register_node<C, GlossyMaterialNode<C>>("GlossyMaterial");
        register_node<C, MixMaterialNode<C>>("MixMaterial");
    }

    AKR_VARIANT void RegisterMaterialNode<C>::register_python_nodes(py::module &m) {
#ifdef AKR_ENABLE_PYTHON
        py::class_<MaterialNode<C>, SceneGraphNode<C>, std::shared_ptr<MaterialNode<C>>>(m, "Material");
        py::class_<DiffuseMaterialNode<C>, MaterialNode<C>, std::shared_ptr<DiffuseMaterialNode<C>>>(m,
                                                                                                     "DiffuseMaterial")
            .def(py::init<>())
            .def_readwrite("color", &DiffuseMaterialNode<C>::color)
            .def("commit", &DiffuseMaterialNode<C>::commit);
        py::class_<EmissiveMaterialNode<C>, MaterialNode<C>, std::shared_ptr<EmissiveMaterialNode<C>>>(
            m, "EmissiveMaterial")
            .def(py::init<>())
            .def_readwrite("color", &EmissiveMaterialNode<C>::color)
            .def_readwrite("double_sided", &EmissiveMaterialNode<C>::double_sided)
            .def("commit", &EmissiveMaterialNode<C>::commit);
#endif
    }

    AKR_RENDER_STRUCT(RegisterMaterialNode)
} // namespace akari